import os
import json
import asyncio
from typing import Dict, List, Optional, Any, TypedDict, Union

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv('docker/.env')

DEFAULT_MODEL = "gpt-4o"
STREAMING = True

INTERVIEW_FIELDS = ["location", "name", "age", "occupation"]


class ResponseStreamingHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.current_message = ""
        print("Agent: ", end="", flush=True)

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
        self.tokens.append(token)
        self.current_message += token

    def get_response(self):
        return AIMessage(content=self.current_message)


class InterviewState(TypedDict):
    qa_dict: Dict[str, Any]
    messages: List[Union[HumanMessage, AIMessage]]
    current_field: Optional[str]
    attempts: Dict[str, int]
    done: bool
    waiting_for_input: bool
    processed_message_count: int


def decide_next_field(state: InterviewState) -> InterviewState:
    """Decide the next field to ask about."""
    qa_dict = state["qa_dict"]

    if state.get("waiting_for_input", False):
        print("DEBUG: Waiting for input in decide_next_field")
        return state

    if (state["messages"] and
        isinstance(state["messages"][-1], HumanMessage) and
            state.get("processed_message_count", 0) < len(state["messages"])):
        print("DEBUG: Have new user message to process")
        return {**state, "waiting_for_input": False}

    if not state["messages"] or state.get("processed_message_count", 0) >= len(state["messages"]):
        print("DEBUG: Finding next field to ask about")
        next_field = None
        for field in INTERVIEW_FIELDS:
            field_data = qa_dict.get(field, {})
            if not field_data.get("response") and field_data.get("attempts", 0) < 3:
                next_field = field
                print(f"DEBUG: Selected next field: {next_field}")
                break

        if next_field is None:
            print("DEBUG: All fields processed, marking as done")
            return {**state, "current_field": None, "done": True, "waiting_for_input": False}
        else:
            return {**state, "current_field": next_field, "waiting_for_input": False}
    return state


def generate_ai_message(state: InterviewState) -> InterviewState:
    """Generate the AI's next message based on the current state."""
    if state.get("waiting_for_input", False):
        print("DEBUG: Waiting for input, not generating message")
        return state

    current_field = state["current_field"]
    qa_dict = state["qa_dict"]
    messages = state["messages"]
    attempts = state["attempts"]

    print(
        f"DEBUG: Generating message - Current field: {current_field}, Done: {state.get('done')}")

    if current_field is None and state["done"]:
        print("DEBUG: Generating closing message")
        collected_info = []
        for field in INTERVIEW_FIELDS:
            field_data = qa_dict.get(field, {})
            response = field_data.get("response", "Not provided")
            collected_info.append(f"{field}: {response}")

        collected = "\n".join(collected_info)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Alex, a real estate agent from Home Realty. You've completed your initial 
            conversation with a potential home seller. Thank the person warmly for their time and information. 
            Mention that you'll use this information to better understand their needs. Offer to help with any 
            questions they might have about the real estate market or selling process. Keep it friendly and not too formal."""),
            ("human", f"Information collected:\n{collected}")
        ])

        llm = ChatOpenAI(model=DEFAULT_MODEL, streaming=STREAMING)
        chain = prompt | llm

        if STREAMING:

            # Create handler and run the chain
            handler = ResponseStreamingHandler()
            _ = response_chain.invoke({}, config={"callbacks": [handler]})
            response = handler.get_response()
            print()
        else:
            response = chain.invoke({})

        return {**state, "messages": messages + [response], "waiting_for_input": True}

    if current_field:
        current_attempts = attempts.get(current_field, 0)
        print(
            f"DEBUG: Asking about {current_field}, attempt {current_attempts + 1}")

        field_questions = {
            "location": "Where are you located? Knowing your area helps me understand the local market conditions.",
            "name": "I didn't catch your name earlier. What should I call you?",
            "age": "If you don't mind me asking, which age group do you fall into? This helps me understand your specific needs.",
            "occupation": "What kind of work do you do? This can help me understand your schedule and availability."
        }

        if current_attempts == 0:
            if not messages:
                system_message = """You are a real estate agent trying to qualify a potential home seller. 
                Introduce yourself politely as a real estate agent named Alex from Home Realty. Explain that
                you're hoping to learn more about their situation to see how you might be able to help them
                with their real estate needs. Be warm, friendly and conversational - like a helpful neighbor
                rather than a formal interviewer. Don't explicitly mention that you're qualifying them as a lead."""
            else:
                system_message = f"""You are a real estate agent named Alex from Home Realty. Ask about the 
                potential client's {current_field} in a conversational way.
                
                Conversation guidelines:
                - This is an ongoing conversation - DO NOT start with "Hey there" or "Hi there" since you've already introduced yourself
                - Avoid starting with phrases like "Certainly!" or "Of course!" since you are initiating 
                  this question, not responding to a request
                - Frame questions in the context of helping them with their real estate needs
                - Use natural transitions between topics that connect to previous exchanges when possible
                - Vary your question formats throughout the conversation to keep it engaging
                - If they've shared something interesting, briefly reference it before your new question
                - Make the conversation feel like a natural flow, not a series of disconnected questions
                - Be genuinely interested in their responses
                - Use a friendly, conversational tone like you're chatting with someone you just met
                - Don't explicitly mention that you're qualifying them as a lead
                """

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human",
                 f"Ask this question: {field_questions.get(current_field)}")
            ])
        else:
            system_message = f"""You are Alex, a real estate agent from Home Realty. This is attempt {current_attempts + 1} 
            to ask about the potential client's {current_field}.
            
            Guidelines:
            - This is an ONGOING conversation - DO NOT start with "Hey there" or "Hi there"
            - Be conversational and natural like you're chatting with a neighbor
            - Acknowledge any previous responses they've given
            - Respect their privacy if they seem reluctant
            - If they've been avoiding the question, try a different approach or explain why this information helps you assist them
            - If they seem interested in something else, briefly engage with that before gently 
              redirecting back to the conversation
            - Maintain a friendly, patient tone without seeming pushy
            - Make your question feel like a natural part of the ongoing conversation
            - Frame questions in terms of how knowing this helps you better assist them with their real estate needs"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human",
                 f"Original question: {field_questions.get(current_field)}")
            ])

        llm = ChatOpenAI(model=DEFAULT_MODEL, streaming=STREAMING)
        chain = prompt | llm

        if STREAMING:
            handler = ResponseStreamingHandler()
            _ = chain.invoke({}, config={"callbacks": [handler]})
            response = handler.get_response()
            print()
        else:
            response = chain.invoke({})

        new_attempts = state["attempts"].copy()
        new_attempts[current_field] = current_attempts + 1

        return {
            **state,
            "messages": messages + [response],
            "attempts": new_attempts,
            "waiting_for_input": True
        }
    print("DEBUG: Reached fallback case in generate_ai_message")
    return {**state, "waiting_for_input": True}


def process_user_input(state: InterviewState) -> InterviewState:
    """Process the user's input and update the state."""
    messages = state["messages"]
    current_field = state["current_field"]
    qa_dict = state["qa_dict"].copy()
    attempts = state["attempts"].copy()
    processed_count = state.get("processed_message_count", 0)

    if len(messages) <= processed_count or not isinstance(messages[-1], HumanMessage):
        print("DEBUG: No new user message to process")
        return {**state, "waiting_for_input": True}

    user_input = messages[-1].content
    print(
        f"DEBUG: Processing user input: {user_input} (message #{len(messages)})")

    if current_field is None:
        print("DEBUG: No current field, but received user input")
        return {
            **state,
            "processed_message_count": len(messages),
            "waiting_for_input": False
        }

    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)

    input_type_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are analyzing a user's response during an interview.
        Classify the input as one of:
        - "answer": User is directly answering the interview question
        - "question": User is asking a question instead of answering
        - "comment": User is making a comment or statement unrelated to the question
        - "clarification": User is asking for clarification about the question
        Be conservative - only classify as "answer" if it's clearly attempting to provide the requested information."""),
        ("human",
         f"Current interview question is about the user's {current_field}.\nUser's response: {user_input}")
    ])

    input_type_chain = input_type_prompt | llm
    input_type = input_type_chain.invoke({})
    input_type_result = input_type.content.strip().lower()
    print(f"DEBUG: Input type classification: {input_type_result}")

    if "question" in input_type_result or "clarification" in input_type_result or "comment" in input_type_result:
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Alex, a real estate agent from Home Realty. The person you're talking with
            has asked a question or made a comment instead of answering your question. Respond appropriately
            to their input, then gently guide them back to the conversation about their home.
            
            Important guidelines:
            - This is an ONGOING conversation - DO NOT start with "Hey there" or "Hi there"
            - If they ask who you are, tell them you're Alex from Home Realty, helping homeowners in their area
            - Respond naturally to what they've said - show you're listening and understanding
            - Maintain a friendly, helpful tone like you're having coffee with a neighbor
            - Avoid abrupt topic changes - use natural transitions to guide back to the conversation
            - Don't be robotic or formulaic in your transitions
            - Acknowledge their question/comment fully before moving on
            - If their question relates to real estate, show your expertise briefly
            - Make your response feel like part of a flowing conversation, not a script
            - Don't explicitly mention qualifying them as a lead"""),
            ("human", f"""User asked/commented: {user_input}
            Current interview question is about: {current_field}
            Generate a helpful response that addresses their input and then returns to asking about their {current_field}.""")
        ])

        llm_response = ChatOpenAI(model=DEFAULT_MODEL, streaming=STREAMING)
        response_chain = response_prompt | llm_response

        if STREAMING:
            handler = ResponseStreamingHandler()
            _ = response_chain.invoke({}, config={"callbacks": [handler]})
            response = handler.get_response()
            print()
        else:
            response = response_chain.invoke({})

        print(f"DEBUG: Generated response to user question/comment")
        return {
            **state,
            "messages": messages + [response],
            "processed_message_count": len(messages),
            "waiting_for_input": True
        }

    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are analyzing a user's response to a question about their {current_field}.
        Classify the response as one of:
        - "given": User clearly provided a value for the field
        - "rejected": User explicitly refused to provide the information
        - "unsure": User is uncertain
        - "incomplete": User's response doesn't answer the question"""),
        ("human", f"User's response: {user_input}")
    ])

    classify_chain = classify_prompt | llm
    classification = classify_chain.invoke({})
    response_type = classification.content.strip().lower()
    print(f"DEBUG: Response classification: {response_type}")

    if "given" in response_type:
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Extract the {current_field} from the user's response.
            Return ONLY the extracted value, nothing else."""),
            ("human", f"User's response: {user_input}")
        ])

        extract_chain = extract_prompt | llm
        extraction = extract_chain.invoke({})
        extracted_value = extraction.content.strip()
        print(f"DEBUG: Extracted value: {extracted_value}")

        if extracted_value:
            qa_dict[current_field] = {
                "response": extracted_value, "attempts": 0}
            print(f"DEBUG: Updated qa_dict: {qa_dict}")
            return {
                **state,
                "qa_dict": qa_dict,
                "attempts": attempts,
                "current_field": None,
                "processed_message_count": len(messages),
                "waiting_for_input": False
            }

    if "reject" in response_type:
        qa_dict[current_field] = {"response": "refused", "attempts": 3}
        print(f"DEBUG: Marked {current_field} as refused")
        return {
            **state,
            "qa_dict": qa_dict,
            "attempts": attempts,
            "current_field": None,
            "processed_message_count": len(messages),
            "waiting_for_input": False
        }

    current_attempts = attempts.get(current_field, 0) + 1
    attempts[current_field] = current_attempts
    print(f"DEBUG: Attempts for {current_field}: {current_attempts}")

    if current_attempts >= 3:
        print(f"DEBUG: Max attempts reached for {current_field}")
        qa_dict[current_field] = {
            "response": None, "attempts": current_attempts}
        return {
            **state,
            "qa_dict": qa_dict,
            "attempts": attempts,
            "current_field": None,
            "processed_message_count": len(messages),
            "waiting_for_input": False
        }

    return {
        **state,
        "attempts": attempts,
        "processed_message_count": len(messages),
        "waiting_for_input": False  # Changed to False to generate a follow-up question
    }


def build_interview_graph():
    """Build the interview graph."""
    builder = StateGraph(InterviewState)

    builder.add_node("decide_next_field", decide_next_field)
    builder.add_node("generate_ai_message", generate_ai_message)
    builder.add_node("process_user_input", process_user_input)

    builder.set_entry_point("decide_next_field")

    builder.add_conditional_edges(
        "decide_next_field",
        lambda state: END if state["done"] else (
            "process_user_input" if (
                state["messages"] and
                isinstance(state["messages"][-1], HumanMessage) and
                state.get("processed_message_count",
                          0) < len(state["messages"])
            ) else (
                END if state["waiting_for_input"] else "generate_ai_message"
            )
        )
    )

    builder.add_conditional_edges(
        "generate_ai_message",
        lambda state: END if state["waiting_for_input"] else "decide_next_field"
    )

    builder.add_conditional_edges(
        "process_user_input",
        lambda state: END if state["waiting_for_input"] else "decide_next_field"
    )

    return builder.compile()


def save_qa_dict(qa_dict: Dict) -> None:
    """Save the QA dictionary to a JSON file."""
    with open("qa_dictionary.json", "w") as f:
        json.dump(qa_dict, f, indent=2)


def load_qa_dict() -> Dict:
    """Load the QA dictionary from a JSON file, or return empty dict if not found."""
    try:
        with open("qa_dictionary.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


async def main():
    """Main function to run the interview."""
    state = {
        "qa_dict": load_qa_dict(),
        "messages": [],
        "current_field": None,
        "attempts": {},
        "done": False,
        "waiting_for_input": False,
        "processed_message_count": 0
    }

    interview_graph = build_interview_graph()
    print("Starting interview agent...")

    try:
        state = interview_graph.invoke(state)
    except Exception as e:
        print(f"Error initializing interview: {e}")
        return

    while not state.get("done", False):
        if state.get("waiting_for_input", True):
            try:
                user_input = input("You: ").strip()
            except KeyboardInterrupt:
                print("\nInterview terminated by user.")
                break
            except Exception as e:
                print(f"Error getting user input: {e}")
                break

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Agent: Thank you for your time. Goodbye!")
                break

            state["messages"].append(HumanMessage(content=user_input))
            state["waiting_for_input"] = False

        try:
            state = interview_graph.invoke(state)
        except Exception as e:
            print(f"Error processing response: {e}")
            break

        try:
            save_qa_dict(state["qa_dict"])
        except Exception as e:
            print(f"Error saving QA dictionary: {e}")

    if "qa_dict" in state:
        save_qa_dict(state["qa_dict"])
    print("Interview completed. QA dictionary saved.")
    

if __name__ == "__main__":
    asyncio.run(main())
