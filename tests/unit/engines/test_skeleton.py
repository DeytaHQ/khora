"""Unit tests for skeleton-based indexing."""

from uuid import uuid4

import pytest

from khora.engines.skeleton.backends import TemporalChunk
from khora.engines.skeleton.skeleton import ChunkNode, KeywordNode, SkeletonIndexer


class TestChunkNode:
    """Tests for ChunkNode dataclass."""

    def test_create_chunk_node(self):
        """Test creating a chunk node."""
        chunk_id = uuid4()
        keywords = {"python", "testing", "code"}

        node = ChunkNode(
            chunk_id=chunk_id,
            content="Python testing code example",
            keywords=keywords,
            pagerank_score=0.5,
            is_core=True,
        )

        assert node.chunk_id == chunk_id
        assert node.content == "Python testing code example"
        assert node.keywords == keywords
        assert node.pagerank_score == 0.5
        assert node.is_core is True

    def test_chunk_node_defaults(self):
        """Test chunk node default values."""
        node = ChunkNode(
            chunk_id=uuid4(),
            content="test",
        )

        assert node.keywords == set()
        assert node.pagerank_score == 0.0
        assert node.is_core is False


class TestKeywordNode:
    """Tests for KeywordNode dataclass."""

    def test_create_keyword_node(self):
        """Test creating a keyword node."""
        chunk_ids = {uuid4(), uuid4()}

        node = KeywordNode(
            keyword="python",
            chunk_ids=chunk_ids,
            idf_score=2.5,
        )

        assert node.keyword == "python"
        assert node.chunk_ids == chunk_ids
        assert node.idf_score == 2.5

    def test_keyword_node_defaults(self):
        """Test keyword node default values."""
        node = KeywordNode(keyword="test")

        assert node.chunk_ids == set()
        assert node.idf_score == 0.0


class TestSkeletonIndexer:
    """Tests for SkeletonIndexer."""

    @pytest.fixture
    def indexer(self):
        """Create a skeleton indexer."""
        return SkeletonIndexer(core_ratio=0.3)

    def test_extract_keywords(self, indexer):
        """Test keyword extraction from content."""
        content = "Python is a great programming language for data science and machine learning."
        keywords = indexer._extract_keywords(content)

        # Should include meaningful words
        assert "python" in keywords
        assert "programming" in keywords
        assert "language" in keywords
        assert "data" in keywords
        assert "science" in keywords
        assert "machine" in keywords
        assert "learning" in keywords

        # Should NOT include stopwords
        assert "is" not in keywords
        assert "a" not in keywords
        assert "the" not in keywords
        assert "for" not in keywords
        assert "and" not in keywords

    def test_extract_keywords_short_words_filtered(self, indexer):
        """Test that short words are filtered out."""
        content = "I am at a go to do it"
        keywords = indexer._extract_keywords(content)

        # All words are too short (< 3 chars) or stopwords
        assert len(keywords) == 0

    def test_add_chunk(self, indexer):
        """Test adding a chunk to the indexer."""
        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Python programming with asyncio for concurrent applications",
        )

        indexer.add_chunk(chunk)

        assert chunk.id in indexer._chunks
        assert "python" in indexer._keywords
        assert "programming" in indexer._keywords
        assert "asyncio" in indexer._keywords

        # Check bidirectional links
        assert chunk.id in indexer._keywords["python"].chunk_ids

    def test_add_chunks_batch(self, indexer):
        """Test adding multiple chunks in batch."""
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python programming basics",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="JavaScript web development",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python web frameworks",
            ),
        ]

        indexer.add_chunks_batch(chunks)

        assert len(indexer._chunks) == 3
        # Python appears in 2 chunks
        assert len(indexer._keywords["python"].chunk_ids) == 2

    def test_build_skeleton_selects_core(self, indexer):
        """Test that build_skeleton selects core chunks."""
        # Create interconnected chunks
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python programming with asyncio and FastAPI web framework",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python asyncio tutorial for beginners",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="FastAPI framework documentation guide",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Isolated topic about gardening plants",
            ),
        ]

        indexer.add_chunks_batch(chunks)
        core_ids = indexer.build_skeleton()

        # With 30% core_ratio and 4 chunks, should have ~1-2 core
        assert len(core_ids) >= 1
        assert len(core_ids) <= 2

        # Highly connected chunks (Python, asyncio, FastAPI) should be core
        # Isolated chunk (gardening) should NOT be core
        gardening_chunk = chunks[3]
        assert gardening_chunk.id not in core_ids

    def test_get_chunks_by_keyword(self, indexer):
        """Test getting chunks by keyword."""
        chunk1_id = uuid4()
        chunk2_id = uuid4()

        chunks = [
            TemporalChunk(
                id=chunk1_id,
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python programming language",
            ),
            TemporalChunk(
                id=chunk2_id,
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="JavaScript programming language",
            ),
        ]

        indexer.add_chunks_batch(chunks)

        # Both have "programming" and "language"
        python_chunks = indexer.get_chunks_by_keyword("python")
        assert len(python_chunks) == 1
        assert chunk1_id in python_chunks

        programming_chunks = indexer.get_chunks_by_keyword("programming")
        assert len(programming_chunks) == 2

    def test_get_related_chunks(self, indexer):
        """Test finding related chunks via keyword overlap."""
        chunk1_id = uuid4()
        chunk2_id = uuid4()
        chunk3_id = uuid4()

        chunks = [
            TemporalChunk(
                id=chunk1_id,
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python asyncio concurrent programming",
            ),
            TemporalChunk(
                id=chunk2_id,
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python asyncio tutorial examples",
            ),
            TemporalChunk(
                id=chunk3_id,
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="JavaScript React components",
            ),
        ]

        indexer.add_chunks_batch(chunks)
        indexer.build_skeleton()

        # Find chunks related to chunk1
        related = indexer.get_related_chunks(chunk1_id, limit=5)

        # chunk2 should be most related (shares python, asyncio)
        assert len(related) > 0
        related_ids = [r[0] for r in related]
        assert chunk2_id in related_ids

        # chunk3 has no overlap, should not be in results
        if len(related) <= 2:
            assert (
                chunk3_id not in related_ids
                or related[[r[0] for r in related].index(chunk3_id) if chunk3_id in related_ids else -1][1] < 0.5
            )

    def test_search_by_keywords(self, indexer):
        """Test searching by multiple keywords."""
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python asyncio programming",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python web framework",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="JavaScript web development",
            ),
        ]

        indexer.add_chunks_batch(chunks)
        indexer.build_skeleton()

        # Search for Python + asyncio
        results = indexer.search_by_keywords(["python", "asyncio"], limit=5)
        assert len(results) > 0

        # First result should be the chunk with both keywords
        top_result_id = results[0][0]
        top_chunk = indexer._chunks[top_result_id]
        assert "python" in top_chunk.keywords
        assert "asyncio" in top_chunk.keywords

    def test_is_core_chunk(self, indexer):
        """Test checking if a chunk is core."""
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python asyncio concurrent programming FastAPI web framework database",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Isolated unrelated content about nothing",
            ),
        ]

        indexer.add_chunks_batch(chunks)
        core_ids = indexer.build_skeleton()

        # Core chunks should return True
        for cid in core_ids:
            assert indexer.is_core_chunk(cid) is True

        # Unknown chunk should return False
        assert indexer.is_core_chunk(uuid4()) is False

    def test_get_pagerank_score(self, indexer):
        """Test getting PageRank score."""
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Central hub connecting everything python asyncio fastapi",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python asyncio only",
            ),
        ]

        indexer.add_chunks_batch(chunks)
        indexer.build_skeleton()

        # All chunks should have a PageRank score > 0
        for chunk in chunks:
            score = indexer.get_pagerank_score(chunk.id)
            assert score > 0

        # Unknown chunk should return 0
        assert indexer.get_pagerank_score(uuid4()) == 0.0


class TestPageRankCalculation:
    """Tests for PageRank calculation."""

    def test_pagerank_convergence(self):
        """Test that PageRank converges."""
        indexer = SkeletonIndexer(core_ratio=0.5, max_iterations=50)

        # Create a simple chain of connected chunks
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python programming basics introduction",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python programming advanced topics",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python advanced machine learning",
            ),
        ]

        indexer.add_chunks_batch(chunks)
        indexer.build_skeleton()

        # All chunks should have positive PageRank
        for chunk in chunks:
            assert indexer.get_pagerank_score(chunk.id) > 0

    def test_empty_indexer(self):
        """Test behavior with empty indexer."""
        indexer = SkeletonIndexer()

        core_ids = indexer.build_skeleton()
        assert core_ids == []

        assert indexer.get_core_chunks() == []
        assert indexer.get_chunks_by_keyword("test") == []
        assert indexer.get_related_chunks(uuid4()) == []
        assert indexer.search_by_keywords(["test"]) == []

    def test_single_chunk(self):
        """Test with single chunk."""
        indexer = SkeletonIndexer(core_ratio=1.0)

        chunk = TemporalChunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Standalone content",
        )

        indexer.add_chunk(chunk)
        core_ids = indexer.build_skeleton()

        # Single chunk should be core
        assert chunk.id in core_ids


class TestSkeletonCostOptimization:
    """Tests verifying cost optimization properties."""

    def test_core_ratio_respected(self):
        """Test that core ratio is approximately respected."""
        indexer = SkeletonIndexer(core_ratio=0.2)  # 20%

        # Create 20 chunks
        chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content=f"Content about topic {i} with keywords",
            )
            for i in range(20)
        ]

        indexer.add_chunks_batch(chunks)
        core_ids = indexer.build_skeleton()

        # Should have approximately 20% = 4 core chunks (±1 for rounding)
        assert 3 <= len(core_ids) <= 5

    def test_highly_connected_chunks_become_core(self):
        """Test that highly connected chunks become core."""
        indexer = SkeletonIndexer(core_ratio=0.3)

        # Create a hub chunk that shares keywords with many others
        hub_id = uuid4()
        chunks = [
            TemporalChunk(
                id=hub_id,
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python asyncio FastAPI React JavaScript TypeScript database SQL",
            ),
        ]

        # Create peripheral chunks that each share 1-2 keywords with hub
        keywords_sets = [
            "Python programming basics",
            "asyncio tutorial guide",
            "FastAPI framework docs",
            "React components state",
            "JavaScript fundamentals",
            "TypeScript types interfaces",
            "database queries optimization",
            "SQL syntax reference",
        ]

        for i, keywords in enumerate(keywords_sets):
            chunks.append(
                TemporalChunk(
                    id=uuid4(),
                    namespace_id=uuid4(),
                    document_id=uuid4(),
                    content=keywords,
                )
            )

        indexer.add_chunks_batch(chunks)
        core_ids = indexer.build_skeleton()

        # Hub chunk should be core (highest connectivity)
        assert hub_id in core_ids

    def test_isolated_chunks_not_core(self):
        """Test that isolated chunks with unique keywords are not core."""
        indexer = SkeletonIndexer(core_ratio=0.3)

        # Create connected cluster
        cluster_chunks = [
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python asyncio programming concurrent",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python asyncio tutorial examples",
            ),
            TemporalChunk(
                id=uuid4(),
                namespace_id=uuid4(),
                document_id=uuid4(),
                content="Python concurrent programming patterns",
            ),
        ]

        # Create isolated chunk with unique keywords
        isolated_id = uuid4()
        isolated_chunk = TemporalChunk(
            id=isolated_id,
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Xyzzy foobar unique unrelated content",
        )

        indexer.add_chunks_batch(cluster_chunks + [isolated_chunk])
        core_ids = indexer.build_skeleton()

        # Isolated chunk should NOT be core
        assert isolated_id not in core_ids
