"""Compatibility wrapper.

You can now run either:
    python ingest.py --documents ./documents --db ./vector_db
or:
    python -m rag.ingest --documents ./documents --db ./vector_db
"""

from rag.ingest import main


if __name__ == "__main__":
    main()
