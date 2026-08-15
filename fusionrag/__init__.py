"""Chunk ids are "{episode}-{index:04d}" (see ingest.py). Adjacent indexes
overlap 50%, so index +-1 within an episode is the same passage."""


def parse_chunk_id(chunk_id):
    episode, index = chunk_id.split("-")
    return episode, int(index)


def adjacent(id_a, id_b):
    ep_a, i_a = parse_chunk_id(id_a)
    ep_b, i_b = parse_chunk_id(id_b)
    return ep_a == ep_b and abs(i_a - i_b) <= 1
