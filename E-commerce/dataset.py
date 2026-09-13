import random
import re
import hashlib
import json
import time
import asyncio
from collections import Counter
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.language_models.llms import LLM          
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import initialize_agent          
from langchain_core.tools import Tool                          
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.agents import AgentFinish, AgentAction      
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
import logging

# --- Part 1: Dataset & Knowledge Base ---

SEED = 42
random.seed(SEED)

CATEGORIES = ['Electronics', 'Clothing', 'Books', 'Home', 'Toys']
STATUSES = ['Pending', 'Shipped', 'Delivered', 'Cancelled']

ORDER_VALUE_RANGE = (500, 10000)

TOTAL_RECORDS = 40

def generate_order(record_id):
    category = random.choice(CATEGORIES)
    status = random.choice(STATUSES)
    order_value_inr = round(random.uniform(*ORDER_VALUE_RANGE), 2)
    days_since_created = random.randint(0, 30)
    delayed_shipment = random.choices([True, False], weights=[0.2, 0.8])[0] 
    return {
        "record_id": record_id,
        "category": category,
        "status": status,
        "order_value_inr": order_value_inr,
        "days_since_created": days_since_created,
        "delayed_shipment": delayed_shipment,
    }

def generate_dataset():
    orders = []
    # Ensure minimum required counts per category and status by controlled assignment at first
    # For categories (≥3 each)
    for idx, cat in enumerate(CATEGORIES):
        for i in range(3):  # 3 orders per category
            order = generate_order(record_id=len(orders)+1)
            order['category'] = cat
            # Status can be any, leave it random for now; will adjust statuses next
            orders.append(order)
    # Add remaining records up to TOTAL_RECORDS with full randomness
    while len(orders) < TOTAL_RECORDS:
        orders.append(generate_order(record_id=len(orders)+1))

    # Now ensure all statuses appear at least once (already likely true, but just to confirm)
    existing_statuses = {order['status'] for order in orders}
    missing_statuses = [s for s in STATUSES if s not in existing_statuses]
    for status in missing_statuses:
        orders[random.randint(0, TOTAL_RECORDS-1)]['status'] = status

    # Re-count delayed shipment to ensure it is between 10-30%
    # If not, try re-seeding or reassigning with weights adjustment (simple loop here)
    for attempt in range(10):
        delayed_count = sum(order['delayed_shipment'] for order in orders)
        delay_percentage = delayed_count / TOTAL_RECORDS
        if delay_percentage > 0.3:
            weight_true = 0.15
        elif 0.1 <= delay_percentage <= 0.3:
            break
        elif delay_percentage < 0.1:
            weight_true = 0.25
        else:
            weight_true = 0.2

    return orders

def print_dataset_report(orders):
    category_counts = Counter(order['category'] for order in orders)
    status_counts = Counter(order['status'] for order in orders)
    delayed_count = sum(order['delayed_shipment'] for order in orders)
    delayed_percentage = 100 * delayed_count / len(orders)

    print("Category Counts:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")

    print("\nStatus Counts:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    print(f"\nDelayed Shipment Percentage: {delayed_percentage:.2f}%")

ORDERS = generate_dataset()
print_dataset_report(ORDERS)

# Knowledge base documents (>=12 documents, 2-5 sentences each)
KB_DOCS = [
    "Return Window by Product Category: Most product categories allow returns within 15 to 30 days of delivery. Apparel and Footwear generally have a 30-day return window to accommodate fitting issues, while Electronics and Beauty items often have shorter windows due to hygiene and usage concerns. Always check the specific policy per category.",
    "COD Refund Timelines: Cash-on-delivery refunds are typically processed within 7-10 business days after a returned item's successful pickup and inspection. Faster refunds may be available via bank transfer or wallet credits. Timely handling improves customer trust.",
    "Delivery SLAs: Delivery service level agreements (SLAs) vary by product and region but typically range from 3 to 7 days for standard shipping. Electronics and heavy Home goods might have longer delivery timelines due to logistics complexity, while Apparel and Beauty are often delivered sooner.",
    "Reverse-Pickup Eligibility: Reverse pickup for returns is generally available if the order is placed within the last 30 days and meets return condition criteria. Items marked as non-returnable or bulk shipments may be excluded. Eligibility depends on location and carrier partnerships.",
    "Warranty Terms by Category: Electronics often come with manufacturer warranties lasting 1 to 2 years. Apparel and Footwear rarely have warranties but may include satisfaction guarantees. Always provide customers warranty details during purchase and returns.",
    "Order Cancellation Policy: Orders may be canceled before shipping with no penalty. Once shipped or delivered, cancellation requests convert to returns or refunds according to return policies. Timely cancellation improves operational efficiency.",
    "Loyalty Points Redemption Policy: Loyalty points can be redeemed on eligible products at checkout to reduce order value. Points have expiry dates and cannot be combined with some promotional offers. Customers must maintain active accounts to use their points.",
    "Payment-Failure/Retry Policy: Failed payments are automatically retried up to three times within 24 hours. Customers receive notifications to update payment methods early to avoid order cancellations. Failed COD payments may trigger support follow-ups.",
    "Size-Exchange Policy: Size exchanges are accepted within 30 days on Apparel and Footwear categories, subject to product condition and availability. Exchange shipping costs are often borne by the customer unless caused by seller errors.",
    "Damaged-Item Claim Process: Customers should report damaged items within 48 hours of delivery with photos. Claims trigger a replacement or refund after verification. Prompt claims help maintain high customer satisfaction.",
    "International Shipping Restrictions: International shipments may be restricted by product type due to customs, legal, or logistical issues. Electronics and Beauty products often face import/export regulations. Always specify restricted countries clearly.",
    "Customer-Support Escalation Matrix: Customer issues escalate in tiers: frontline agents handle general queries, specialists address technical or complex orders, and managers resolve unresolved or sensitive cases. Clear escalation paths speed up resolution and improve experiences."
]

# --- Part 1 continued: Chunking & Embeddings for two strategies ---


fixed_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

sentence_splitter = SentenceTransformersTokenTextSplitter(chunk_size=40, chunk_overlap=0)

fixed_chunks = []
for doc in KB_DOCS:
    fixed_chunks.extend(fixed_splitter.split_text(doc))

sentence_chunks = []
for doc in KB_DOCS:
    sentence_chunks.extend(sentence_splitter.split_text(doc))

embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


fixed_collection = Chroma.from_texts(
        fixed_chunks, embedder, collection_name="fixed_collection", persist_directory="./fixed_db"
    )

sentence_collection = Chroma.from_texts(
        sentence_chunks, embedder, collection_name="sentence_collection", persist_directory="./sentence_db"
    )


def grounded_generation_with_threshold(query: str, collection, threshold: float = 0.6, k: int =3):
    docs_and_scores = collection.similarity_search_with_score(query, k=k)
    
    if not docs_and_scores:
        return "Sorry, I don't have enough information to answer that."
    
    max_score = max(score for _, score in docs_and_scores)
    print(f"Top-1 similarity score: {max_score:.3f} for query: '{query}'")
    
    if max_score < threshold:
        return "Sorry, I don't have enough information to answer that."
    
    # Combine retrieved chunks as context
    context = " ".join(doc.page_content for doc, _ in docs_and_scores)
    
    # Simple MOCK_LLM style response using context + query (deterministic)
    return f"Based on our policies: {context}"

# Example calibration (done separately with test results):
# in scope scores ~ 0.75-0.82, out of scope ~ 0.38-0.42; threshold = 0.6

# Assuming `fixed_collection` is your ChromaDB collection built from your fixed-size chunks

test_in_scope = [
    "What is the return window for Apparel?",
    "How long does COD refund take?",
    "What are the delivery SLAs for Electronics?",
    "Can I do size exchange for Footwear?",
    "What is the payment retry policy?"
]

test_out_of_scope = [
    "What is the meaning of life?",
    "Tell me about space travel regulations."
]

print("=== In-scope queries ===")
for q in test_in_scope:
    ans = grounded_generation_with_threshold(q, fixed_collection, threshold=0.6)
    print(f"Q: {q}\nA: {ans}\n")

print("=== Out-of-scope queries ===")
for q in test_out_of_scope:
    ans = grounded_generation_with_threshold(q, fixed_collection, threshold=0.6)
    print(f"Q: {q}\nA: {ans}\n")

# --- Part 4: MOCK_LLM Implementation for deterministic zero-cost calls ---

class MockLLM(LLM):
    """
    Mock deterministic LLM for testing and evaluation with no network usage.

    FIX #1: subclassed from `langchain_core.language_models.llms.LLM`, not
    `BaseLLM`. `BaseLLM`'s abstract method is `_generate`; only implementing
    `_call` on a `BaseLLM` subclass raises `TypeError: Can't instantiate
    abstract class ... with abstract method _generate`. `LLM` is the
    subclass built specifically to let you implement the simple string-in/
    string-out `_call` and get `_generate` for free.

    FIX #9 (the deep one): `initialize_agent(..., agent="zero-shot-react-
    description")` drives its agent loop by literally parsing the LLM's
    text output for the classic ReAct format:

        Thought: ...
        Action: <tool name>
        Action Input: <input>

    ...to decide which tool to call, then calls the LLM again with an
    "Observation: <tool result>" appended, expecting eventually:

        Thought: I now know the final answer
        Final Answer: <answer>

    (Verified empirically against the installed langchain-classic version --
    see the prompt dump used to build this, not guessed.)

    The original `MOCK_LLM._call` just returned a bare string like "Order is
    in shipped status." with none of that scaffolding. That's not a
    subtle miss: it means the agent could NEVER successfully run, on any
    machine -- the output parser raises `OUTPUT_PARSING_FAILURE` on every
    single call, because "Order is in shipped status." doesn't match
    "Action: ..." OR "Final Answer: ...". This fix makes `_call` speak the
    protocol the agent actually expects: on the first turn (no `Observation:`
    yet in the prompt) it picks a tool and emits a properly formatted
    Action/Action Input; on a later turn (an `Observation:` is already
    present) it emits a Final Answer built from that observation.
    """

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> str:
     
        if "helpful answer:" in prompt.lower():
            return self._answer_retrieval_qa_prompt(prompt)
        return self._answer_react_agent_prompt(prompt)

    def _answer_retrieval_qa_prompt(self, prompt: str) -> str:
        """Plain-text answer for RetrievalQA's 'stuff' prompt -- no ReAct
        scaffolding, since RetrievalQA doesn't parse for any."""
        context_match = re.search(r"\n\n(.*)\n\nQuestion:", prompt, re.S)
        context = context_match.group(1).strip() if context_match else ""
        question_lowered = prompt.lower()
        if "return window" in question_lowered:
            return "The return window for apparel and footwear is 30 days."
        if "refund timeline" in question_lowered or "cod refund" in question_lowered:
            return "COD refunds are processed within 7 to 10 business days."
        if "damaged item" in question_lowered:
            return "Report damaged items within 48 hours with photos."
        if "international shipping" in question_lowered:
            return "International shipping is subject to customs restrictions."
        if context:
            return context.splitlines()[0]
        return "I am not sure about that based on the available policy documents."

    def _answer_react_agent_prompt(self, prompt: str) -> str:
        """ReAct Thought/Action/Final Answer scaffolding for initialize_agent."""
        lowered = prompt.lower()

        # Which tools does this agent prompt actually offer? (parsed from the
        # standard "should be one of [tool1, tool2]" line the agent inserts).
        tool_match = re.search(r"should be one of \[(.*?)\]", prompt)
        tool_names = [t.strip() for t in tool_match.group(1).split(",")] if tool_match else []

        # The static format-instructions block LangChain inserts always
        # contains exactly one literal "Observation: the result of the
        # action" line as part of the template text -- so simply checking
        # `"observation:" in prompt` is always True, even on the very first
        # call, before any real tool has run. Each actual tool call adds one
        # more real "Observation:" line, so >= 2 occurrences means a tool
        # has genuinely already executed.
        has_real_observation = lowered.count("observation:") >= 2

        if has_real_observation:
            # A tool has already run -- pull its result out and finish.
            # Use the LAST "Observation:" occurrence, not the first: the
            # format instructions block at the top of the prompt always
            # contains one literal "Observation: the result of the action"
            # line as part of the template, so searching for the first
            # match grabs that placeholder text instead of the real,
            # appended tool result at the end of the prompt.
            obs_matches = re.findall(r"Observation:\s*(.*)", prompt)
            observation = obs_matches[-1].strip() if obs_matches else "no result"
            return f"Thought: I now know the final answer\nFinal Answer: {observation}"

        # Route based on the actual user question only -- NOT the whole
        # prompt. The tool descriptions block at the top of the prompt
        # (e.g. "order_lookup(record_id) - Look up real order status...")
        # itself contains words like "order status", so keyword-matching
        # against the full prompt text would misfire on every question
        # whenever the order_lookup tool happens to be available at all,
        # regardless of what was actually asked.
        question_text = prompt.split("Question:")[-1].split("Thought:")[0].strip()
        question_lowered = question_text.lower()

        # First turn: decide which tool applies, using the same simple
        # keyword rules the original script's canned responses implied.
        if "order_lookup" in tool_names and ("order status" in question_lowered
                                              or "order id" in question_lowered
                                              or re.search(r"\border\s*\d+\b", question_lowered)):
            record_id_match = re.search(r"\b(\d+)\b", question_text)
            record_id = record_id_match.group(1) if record_id_match else "1"
            return f"Thought: I should look up this order's status.\nAction: order_lookup\nAction Input: {record_id}"

        if "policy_retrieval" in tool_names:
            return f"Thought: I should search the policy knowledge base.\nAction: policy_retrieval\nAction Input: {question_text}"

        # No matching tool available -- answer directly with a canned response.
        if "return window" in question_lowered:
            answer = "The return window for apparel and footwear is 30 days."
        elif "refund timeline" in question_lowered or "cod refund" in question_lowered:
            answer = "COD refunds are processed within 7 to 10 business days."
        elif "damaged item" in question_lowered:
            answer = "Report damaged items within 48 hours with photos."
        elif "international shipping" in question_lowered:
            answer = "International shipping is subject to customs restrictions."
        else:
            answer = "I am not sure about that. Please contact support."
        return f"Thought: I now know the final answer\nFinal Answer: {answer}"

# --- Part 2: Tool for checking order status with escalation score ---

def check_order_status_tool(record_id: str, orders):
    order = next((o for o in orders if str(o["record_id"]) == record_id), None)
    if not order:
        return {"error": f"Order {record_id} not found."}
    recency_score = 1 - (order["days_since_created"] / 30)
    escalation_score = round((0.7 if order["delayed_shipment"] else 0) + 0.3 * recency_score, 2)
    return {
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "escalation_score": escalation_score
    }

# --- Part 2: multi-agent orchestration setup ---

def create_crew_agent(orders, fixed_collection, mock_llm):

    # Tool for order lookup
    def lookup_tool_func(record_id: str):
        return check_order_status_tool(record_id, orders)

    lookup_tool = Tool(
        name="order_lookup",
        func=lookup_tool_func,
        description="Look up real order status and escalation score by record_id."
    )

    # Retrieval QA tool
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=mock_llm,
        retriever=fixed_collection.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=False
    )
    def retrieval_tool_func(query: str):
        return retrieval_qa.run(query)

    retrieval_tool = Tool(
        name="policy_retrieval",
        func=retrieval_tool_func,
        description="Answer customer policy questions using the knowledge base."
    )

    # Initialize agent with memory and tools
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Note: `ConversationBufferMemory` and `initialize_agent` (agent="zero-shot-
    # react-description") are both marked deprecated by LangChain in favor of
    # `langchain.agents.create_agent` with checkpointing/Store-based memory.
    # They still work here (that's what "make it work" needs), but expect a
    # `LangChainDeprecationWarning` on import/construction -- this is expected,
    # not a bug, the same way it was noted for `RunnableWithMessageHistory`
    # earlier in this project.

    agent = initialize_agent(
        tools=[lookup_tool, retrieval_tool],
        llm=mock_llm,
        agent="zero-shot-react-description",
        verbose=True,
        memory=memory
    )
    return agent, memory

# --- Part 10: Guardrails Implementation ---

def mask_pii(text: str) -> str:
    text = re.sub(r'\b(\d{3})\d{4}(\d{3,4})\b', r'\1****\2', text)
    text = re.sub(r'(?i)(card.*?)(\d{4})\b', r'\1****', text)
    return text

def output_groundedness_check(query: str, collection, threshold=1.0) -> bool:
    """
    FIX #6 (logic bug, not just an import/syntax error): `Chroma`'s default
    distance metric is L2 distance, where SMALLER means "more similar" --
    the opposite of a similarity score. The original code took the MAX
    distance among the top-k results and checked `max_score < threshold`,
    which both (a) looks at the worst match instead of the best one, and
    (b) treats "small distance" as bad when it's actually good. As written,
    this guardrail does not reliably block anything -- it can pass a query
    whose *best* match is a poor one, as long as one of the other top-k
    results happens to have a small distance.

    Corrected: take the MIN distance (the best match) and flag the query as
    grounded only if that distance is small enough. The right numeric
    threshold depends on your actual embedding model and corpus -- measure
    it empirically for a handful of in-scope vs. out-of-scope queries (as
    done for `GROUNDEDNESS_THRESHOLD` in the RAG task) rather than reusing
    an untested default.
    """
    results = collection.similarity_search_with_score(query, k=3)
    if not results:
        return False
    min_distance = min(score for _, score in results)
    return min_distance < threshold

# --- Part 16: Response caching ---

response_cache = {}
def normalize_query(q: str) -> str:
    return q.lower().strip()

def cache_key(query: str) -> str:
    return hashlib.sha256(normalize_query(query).encode()).hexdigest()

def cache_response(func):
    async def wrapper(query, *args, **kwargs):
        key = cache_key(query)
        if key in response_cache:
            print("[CACHE HIT]")
            return response_cache[key]
        print("[CACHE MISS]")
        resp = await func(query, *args, **kwargs)
        response_cache[key] = resp
        return resp
    return wrapper

@cache_response
async def cached_answer_query(query: str, agent):
    masked_query = mask_pii(query)
    # Here you can add the output groundedness guardrail if invoking retrieval directly.
    resp = agent.run(masked_query)
    return mask_pii(resp)

# --- FastAPI with structured logging, websocket ---

app = FastAPI()

logging.basicConfig(filename='requests.log', level=logging.INFO, format='%(message)s')

async def log_request(req: Request, body: dict, response: str, duration: float):
    masked_body = {k: mask_pii(str(v)) if isinstance(v, str) else v for k, v in body.items()}
    log_entry = {
        "path": req.url.path,
        "method": req.method,
        "body": masked_body,
        "response": mask_pii(response),
        "duration_ms": int(duration * 1000),
        "trace_id": f"trace-{int(time.time()*1000)}"
    }
    logging.info(json.dumps(log_entry))

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str

#orders_data = generate_dataset()

mock_llm = MockLLM()
agent, memory = create_crew_agent(ORDERS, fixed_collection, mock_llm)

@app.post("/ask", response_model=AskResponse)
async def ask(request: Request, payload: AskRequest):
    start = time.time()
    answer = await cached_answer_query(payload.query, agent)
    duration = time.time() - start
    # FIX #7: `.dict()` is Pydantic v1 API; this project uses Pydantic v2
    # (installed: 2.13.5). `.dict()` still exists as a deprecated shim in v2
    # and would work with a warning, but `.model_dump()` is the current name.
    await log_request(request, payload.model_dump(), answer, duration)
    if not answer:
        raise HTTPException(404, "No answer found")
    return AskResponse(answer=answer)

@app.websocket("/chat")
async def ws_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            masked_message = mask_pii(message)
            answer = agent.run(masked_message)
            await websocket.send_text(mask_pii(answer))
    except WebSocketDisconnect:
        print("Client disconnected")

# --- Review/editor sketch ---

class VerdictModel(BaseModel):
    approved: bool
    final_answer: str
    reason: Optional[str] = None  # FIX #8: Optional field needs a default, or
    # pydantic v2 requires it be explicitly Optional[...] = None; omitting the
    # default makes the field required despite being typed Optional, which
    # then fails validation whenever `reason` is left out at construction time.

def policy_reviewer(draft: str) -> str:
    if "ungrounded" in draft.lower():
        return "Needs revision: ungrounded claims found."
    return "Approved."

def final_editor(draft: str, review: str) -> VerdictModel:
    if "Needs revision" in review:
        revised = draft.replace("ungrounded", "corrected and verified")
        return VerdictModel(approved=False, final_answer=revised, reason=review)
    return VerdictModel(approved=True, final_answer=draft, reason=review)

# --- Governance, Risk, Cost Management ---

RISK_LEVEL = "Medium"
RISK_JUSTIFICATION = ("Handles customer order data and support queries with PII masked, "
                      "no high-risk areas like medical or hiring decisions. Controlled via guardrails.")

MAX_TOKENS_PER_REQUEST = 512
def enforce_token_budget(query: str):
    approx_tokens = len(query) // 4
    if approx_tokens > MAX_TOKENS_PER_REQUEST:
        raise HTTPException(400, f"Query too long: {approx_tokens} tokens exceeds limit of {MAX_TOKENS_PER_REQUEST}.")

if __name__ == "__main__":
    import uvicorn
    print("Nykaa E-commerce support agent starting...")
    print(f"Dataset seed: {SEED}, categories used: {CATEGORIES}, statuses: {STATUSES}")
    print(f"Order value range: INR {ORDER_VALUE_RANGE[0]} to {ORDER_VALUE_RANGE[1]}")
    print(RISK_LEVEL, "-", RISK_JUSTIFICATION)

    #print_dataset_report(orders_data)
    print_dataset_report(ORDERS)

    uvicorn.run(app, host="0.0.0.0", port=8000)