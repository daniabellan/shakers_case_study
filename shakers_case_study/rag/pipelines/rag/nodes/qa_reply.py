import time

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.rag.pipelines.rag.enums.sentiment_map import SENTIMENT_TONES
from shakers_case_study.rag.pipelines.rag.nodes.get_user_history import get_user_history
from shakers_case_study.rag.pipelines.rag.nodes.log_llm_metrics import log_llm_metrics
from shakers_case_study.rag.pipelines.rag.nodes.recommended_resources_personalized import (
    recommend_resources_personalized,
)
from shakers_case_study.rag.pipelines.rag.prompts.prompts import (
    COMPANY_QA_PROMPT,
    NO_RESOURCES_FOUND_PROMPT,
)


def qa_reply(state: MyStateSchema, config: RunnableConfig) -> MyStateSchema:
    """
    Processes the user's question and generates an answer using an LLM with retrieval and
    personalized resource recommendations.

    Steps:
    - Retrieves user chat history.
    - Generates personalized resource recommendations.
    - Performs similarity search on vectorstore for relevant documents.
    - Constructs an LLM prompt with company info or fallback if no documents found.
    - Invokes the LLM to get the answer.
    - Appends the answer and recommendations to the conversation state.
    - Logs LLM usage metrics.

    Args:
        state (MyStateSchema): Current pipeline state including messages and sentiment.
        config (RunnableConfig): Configuration with LLM, vectorstore, and other resources.

    Returns:
        MyStateSchema: Updated state including the assistant's reply and updated metrics.
    """
    start_time = time.time()
    user_question = state.messages[-1]["content"]
    vectorstore = config["configurable"]["vectorstore"]
    llm = config["configurable"]["llm"]

    # Map sentiment to tone, default to neutral if unknown
    sentiment_tone = SENTIMENT_TONES.get(state.sentiment, SENTIMENT_TONES["neutral"])

    # Retrieve historical user messages for personalization and context
    user_history = get_user_history(config)

    # Generate personalized resource recommendations based on user history and question
    recommendations_payload = recommend_resources_personalized(
        user_history, state, config, user_question
    )
    state = recommendations_payload["state"]
    recommendations = recommendations_payload["recommendations"]

    # Search for relevant documents in the vectorstore with similarity scoring
    resources = vectorstore.similarity_search_with_score(query=user_question)

    if resources:
        # Format retrieved documents into a structured string for the prompt
        company_info = "\n".join(
            f"""<document>  # noqa: E501
    <source>{doc[0].metadata.get('source_file', 'Unknown Resource').replace('_', ' ').title()}</source>
    <content>{doc[0].page_content.strip()}</content>
    </document>"""
            for doc in resources
        )
        prompt_template = COMPANY_QA_PROMPT
    else:
        # Use fallback prompt if no relevant documents are found
        company_info = ""
        prompt_template = NO_RESOURCES_FOUND_PROMPT

    # Format the full prompt with company info and sentiment tone
    full_prompt_str = prompt_template.format(
        company_info=company_info,
        sentiment_tone=sentiment_tone,
    )

    prompt = PromptTemplate(
        input_variables=["user_question", "previous_context"],
        template=full_prompt_str,
    )

    # Format prompt with user question (previous_context not used here explicitly)
    formatted_prompt = prompt.format(user_question=user_question)

    messages = [HumanMessage(content=formatted_prompt)]

    # Invoke the LLM to generate the answer
    llm_response = llm.invoke(messages)

    # Format recommendations as a list in the response
    recommendations_text = "\n\n".join(
        f"- {rec['source_file']}: {rec['explanation']}" for rec in recommendations
    )

    final_response = (
        f"{llm_response.content}\n\n"
        f"\n\nRecommendations:\n{recommendations_text if recommendations_text else 'No additional recommendations available.'}"  # noqa: E501
    )

    # Update state with assistant's reply and current node
    state.current_node = "question_answer"
    state.messages.append({"role": "assistant", "content": final_response})

    # Log latency and token usage metrics for this LLM call
    state = log_llm_metrics(state, "qa_reply", start_time)

    return state
