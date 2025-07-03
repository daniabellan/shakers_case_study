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


# Nodo que procesa input usuario y genera respuesta con LLM
def qa_reply(
    state: MyStateSchema,
    config: RunnableConfig,
):
    start_time = time.time()
    user_question = state.messages[-1]["content"]
    vectorstore = config["configurable"]["vectorstore"]
    llm = config["configurable"]["llm"]

    sentiment_tone = SENTIMENT_TONES.get(state.sentiment, SENTIMENT_TONES["neutral"])

    # Get user historic chat
    user_history = get_user_history(config)

    # RECOMMENDATION SYSTEM
    recommendations_payload = recommend_resources_personalized(
        user_history,
        state,
        config,
        user_question,
    )
    state = recommendations_payload["state"]
    recommendations = recommendations_payload["recommendations"]

    # GENERATE USER ANSWER
    # Retrieval
    resources = vectorstore.similarity_search_with_score(query=user_question)

    # Fallback: verificar si hay documentos
    if resources:
        company_info = "\n".join(
            f"""<document>  # noqa: E501
    <source>{doc[0].metadata.get('source_file', 'Unknown Resource').replace('_', ' ').title()}</source>
    <content>{doc[0].page_content.strip()}</content>
    </document>"""
            for doc in resources
        )
        full_prompt_str = COMPANY_QA_PROMPT
    else:
        # Prompt alternativo si no se encuentran recursos relevantes
        company_info = ""
        full_prompt_str = NO_RESOURCES_FOUND_PROMPT

    full_prompt_str = full_prompt_str.format(
        company_info=company_info,
        sentiment_tone=sentiment_tone,
    )

    prompt = PromptTemplate(
        input_variables=["user_question", "previous_context"],
        template=full_prompt_str,
    )
    formatted_prompt = prompt.format(
        user_question=user_question,
    )

    messages = [HumanMessage(content=formatted_prompt)]
    llm_response = llm.invoke(messages)

    recommendations_text = "\n\n".join(
        f"- {rec['source_file']}: {rec['explanation']}" for rec in recommendations
    )

    final_response = (
        f"{llm_response.content}\n\n"
        f"---\nRecommendations:\n{recommendations_text if recommendations_text else 'No additional recommendations available.'}"  # noqa: E501
    )

    state.current_node = "question_answer"
    state.messages.append({"role": "assistant", "content": final_response})

    state = log_llm_metrics(state, "qa_reply", start_time)
    return state
