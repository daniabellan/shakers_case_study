from functools import lru_cache


def make_cached_embedder(embedder):
    @lru_cache(maxsize=1000)
    def embed(text: str):
        return embedder.embed_query(text)

    return embed
