#  Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
#  Permission is hereby granted, free of charge, to any person
#  obtaining a copy of this software and associated documentation
#  files (the "Software"), to deal in the Software without
#  restriction, subject to the conditions in the full MIT License.
#  The Software is provided "as is", without warranty of any kind.

from pydantic import BaseModel, Field, conlist


class RerankRequest(BaseModel):
    query: str = Field(..., description="The search query")
    documents: conlist(str, min_length=1) = Field(
        ..., description="List of documents to rerank"
    )
    batch_size: int = Field(32, description="Batch size for the model")
    top_n: int | None = Field(
        None,
        description="Number of top results to return. If not specified, uses DEFAULT_TOP_N env var or returns all results.",
        alias="top_k",
    )


class RerankResult(BaseModel):
    index: int = Field(
        ..., description="The index of the document in the original list"
    )
    relevance_score: float = Field(
        ..., description="The relevance score of the document"
    )


class RerankUsage(BaseModel):
    total_tokens: int = Field(..., description="Total number of tokens processed")


class RerankResponse(BaseModel):
    model: str = Field(..., description="The model used for reranking")
    object: str = Field(default="list", description="The object type")
    results: list[RerankResult] = Field(..., description="List of reranked results")
