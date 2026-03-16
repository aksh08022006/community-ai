import logging
import asyncio

import transformers
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

pipeline = None
_pipeline_lock = asyncio.Lock()

# Pre-quantized GPTQ model for optimized inference
MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"


async def init_local_llm(max_retries: int = 3, initial_delay: float = 5.0) -> bool:
    """
    Lazily initialize the local LLM pipeline with simple exponential backoff.
    Returns True on success, False on failure.
    """
    global pipeline

    if pipeline is not None:
        return True

    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Initializing pre-quantized pipeline for '%s' (attempt %d/%d)...",
                MODEL_ID,
                attempt,
                max_retries,
            )
            pipeline = await asyncio.to_thread(
                transformers.pipeline,
                "text-generation",
                MODEL_ID,
                device_map="auto",
            )
            logger.info("Local LLM pipeline initialized successfully.")
            return True
        except Exception as e:
            logger.error(
                "Failed to initialize local LLM pipeline on attempt %d/%d: %s",
                attempt,
                max_retries,
                e,
                exc_info=True,
            )
            if attempt < max_retries:
                logger.info("Retrying in %.1f seconds...", delay)
                await asyncio.sleep(delay)
                delay *= 2

    logger.error("Exhausted retries; local LLM pipeline could not be initialized.")
    pipeline = None
    return False


async def get_llm_response(prompt: str) -> str | None:
    global pipeline
    if pipeline is None:
        initialized = await init_local_llm()
        if not initialized:
            logger.error("Cannot get LLM response because the pipeline could not be initialized.")
            return None
    try:
        async with _pipeline_lock:
            def _run_inference():
                messages = [{"role": "user", "content": prompt}]
                outputs = pipeline(messages, max_new_tokens=512, do_sample=False)
                
                # Handle both string and chat-history output structures
                generated_text = ""
                try:
                    if isinstance(outputs, list) and outputs:
                        first_output = outputs[0]
                        if isinstance(first_output, dict) and "generated_text" in first_output:
                            raw_generated = first_output["generated_text"]
                            
                            # Case 1: generated_text is already a string
                            if isinstance(raw_generated, str):
                                generated_text = raw_generated
                            # Case 2: generated_text is a list (e.g., chat-history)
                            elif isinstance(raw_generated, list) and raw_generated:
                                last_item = raw_generated[-1]
                                if isinstance(last_item, dict) and "content" in last_item:
                                    generated_text = last_item["content"]
                                else:
                                    generated_text = str(last_item)
                            else:
                                generated_text = str(raw_generated)
                        else:
                            logger.warning("Unexpected output structure: missing 'generated_text' key.")
                            generated_text = str(first_output)
                    else:
                        logger.warning("Unexpected pipeline output format or empty output.")
                        generated_text = str(outputs)
                except Exception as extract_err:
                    logger.error(
                        f"Failed to extract generated_text from pipeline output: {extract_err}",
                        exc_info=True,
                    )
                    generated_text = str(outputs)

                json_start_index = generated_text.find("{")
                json_end_index = generated_text.rfind("}")

                if json_start_index != -1 and json_end_index != -1:
                    return generated_text[json_start_index : json_end_index + 1]
                logger.warning(f"Could not find JSON in LLM response: {generated_text}")
                return generated_text

            return await asyncio.to_thread(_run_inference)
    except Exception as e:
        logger.error(f"An error occurred while running the local LLM pipeline: {e}", exc_info=True)
        return None