import os
import logging


logger = logging.getLogger(__name__)

def login(token):
    from huggingface_hub import login
    import time

    while True:
        try:
            login(token)
            logger.info('HF login ok.')
            break
        except Exception as e:
            logger.warning(f'HF login failed: {e}. Retrying...')
            time.sleep(0.5)


hf_token = os.environ.get('HF_TOKEN', None)

if hf_token is not None:
    login(hf_token)
