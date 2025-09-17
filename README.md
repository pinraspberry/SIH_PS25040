# FloatChat - Argo NetCDF RAG Pipeline

> **Smart India Hackathon (SIH) 2024 Submission**

FloatChat is an AI-powered system that makes complex oceanographic ARGO data accessible through natural language queries. This prototype demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline that processes NetCDF files, creates vector embeddings, and enables semantic search with LLM-powered answer generation.

## 🌊 About the Project

ARGO floats collect critical oceanographic data including temperature, salinity, and pressure measurements at various depths across the world's oceans. FloatChat transforms this complex scientific data into an interactive, queryable knowledge base that researchers and decision-makers can explore using natural language.

### Key Features

- **🔄 Data Processing**: Automated NetCDF file ingestion and profile extraction
- **🧠 AI-Powered Search**: Vector embeddings with semantic similarity search
- **💬 Natural Language Queries**: Ask questions about ocean data in plain English
- **📊 Structured Storage**: PostgreSQL database with pgvector for efficient retrieval
- **🔀 Flexible AI Backend**: Support for both local (Llama) and cloud (Gemini) models
- **📈 Scalable Architecture**: Designed for production deployment and large datasets

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NetCDF Files  │───▶│  Data Ingestion  │───▶│ Vector Database │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Profile Summary │    │ Embedding Service│    │ Semantic Search │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Local Models   │    │   Gemini API     │    │  RAG Generation │
│   (Llama)       │    │   (Fallback)     │    │   & Answers     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **PostgreSQL**: 17+ with pgvector extension
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space for models and data

### Python Dependencies
```
xarray>=2023.1.0
netcdf4>=1.6.0
sentence-transformers>=2.2.0
psycopg2-binary>=2.9.0
pgvector>=0.2.0
llama-cpp-python>=0.2.0
google-generativeai>=0.3.0
numpy>=1.21.0
pandas>=1.5.0
```

### Optional Requirements
- **Gemini API Key**: For cloud-based embeddings and generation
- **Local Llama Models**: For offline AI capabilities
- **CUDA Support**: For GPU acceleration (optional)

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd floatchat-argo-rag

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Database Setup

```bash
# Test PostgreSQL connection
python test_database_connection.py

# Setup vector capabilities
python scripts/run_vector_migration.py

# Validate setup
python verify_rag_setup.py
```

### 3. Configure Environment

Create or update `.env` file:
```bash
# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=argo_sih
DATABASE_USER=postgres
POSTGRES_PASSWORD=your_password

# AI Configuration (Optional)
GEMINI_API_KEY=your_gemini_api_key
EMBEDDING_MODEL=sentence-transformers
EMBEDDING_DIMENSION=384
MAX_SEARCH_RESULTS=10
```

### 4. Test with Sample Data

```bash
# Process sample NetCDF file (when ready)
python ingest_argo.py data/19991021_prof.nc

# Run interactive queries (when ready)
python query_demo.py
```

## 📁 Project Structure

```
floatchat-argo-rag/
├── 📁 .kiro/
│   └── specs/                    # Project specifications
├── 📁 backend/
│   ├── models/                   # Data models
│   ├── utils/                    # Utility functions
│   └── requirements.txt          # Python dependencies
├── 📁 config/
│   └── database.py              # Database configuration
├── 📁 data/
│   ├── uploads/                 # File upload directory
│   ├── 19991021_prof.nc        # Sample Argo NetCDF file
│   └── ar_index_global_meta.txt.gz
├── 📁 migrations/
│   └── 001_add_vector_capabilities.sql
├── 📁 scripts/
│   ├── run_vector_migration.py  # Database migration
│   └── validate_vector_schema.py
├── 📄 .env                      # Environment variables
├── 📄 database_schema.sql       # Base database schema
├── 📄 test_database_connection.py
├── 📄 verify_rag_setup.py
└── 📄 README.md
```

## 🔧 Usage Instructions

### Data Ingestion

1. **Prepare NetCDF Files**: Place your Argo NetCDF files in the `data/` directory
2. **Run Ingestion**: Execute the ingestion pipeline to process and store data
3. **Verify Storage**: Check that profiles and embeddings are correctly stored

```bash
# Example ingestion command (when implemented)
python ingest_argo.py data/your_netcdf_file.nc --batch-size 1000
```

### Querying Data

1. **Natural Language Queries**: Ask questions about ocean data in plain English
2. **Semantic Search**: Find relevant profiles using vector similarity
3. **AI-Generated Answers**: Get comprehensive responses with context

```bash
# Example query command (when implemented)
python query_demo.py
# Then enter queries like:
# "Show me temperature profiles near the equator"
# "What are the salinity levels in the North Atlantic?"
```

### Database Management

```bash
# Test database connection
python test_database_connection.py

# Run database migrations
python scripts/run_vector_migration.py

# Validate vector schema
python scripts/validate_vector_schema.py

# Complete system verification
python verify_rag_setup.py
```

## 📊 Implementation Status

### ✅ Completed Components
- [x] Database schema with pgvector extension
- [x] Environment configuration and setup
- [x] Vector migration scripts and validation
- [x] Project structure and documentation
- [x] Database connection testing utilities

### 🚧 In Development
- [ ] Embedding service (local + cloud fallback)
- [ ] NetCDF profile processor and summarizer
- [ ] Vector database manager for profiles
- [ ] Complete ingestion pipeline script

### 📋 Planned Features
- [ ] Semantic search and query processing
- [ ] RAG generation with Llama integration
- [ ] Interactive query demonstration interface
- [ ] Performance optimization and monitoring
- [ ] Web-based user interface
- [ ] Real-time data streaming capabilities

## 🛠️ Development Timeline

**Phase 1 (Current)**: Core Infrastructure ✅
- Database setup and vector capabilities
- Environment configuration
- Basic project structure

**Phase 2 (Next 2-3 days)**: Data Processing 🚧
- NetCDF ingestion pipeline
- Embedding computation and storage
- Profile summarization

**Phase 3 (Days 4-5)**: AI Integration 📋
- Semantic search implementation
- RAG generation system
- Query interface development

**Phase 4 (Days 6-7)**: Testing & Demo 📋
- End-to-end testing
- Performance optimization
- Demo preparation and documentation

## 🔍 Available Scripts

### Database Management
- `test_database_connection.py` - Test PostgreSQL connectivity
- `scripts/run_vector_migration.py` - Add pgvector capabilities
- `scripts/validate_vector_schema.py` - Validate vector schema
- `verify_rag_setup.py` - Complete system verification

### Data Processing (Coming Soon)
- `ingest_argo.py` - NetCDF ingestion pipeline
- `query_demo.py` - Interactive query interface

### Configuration
- `.env` - Environment variables
- `config/database.py` - Database configuration
- `backend/requirements.txt` - Python dependencies

## 🐛 Troubleshooting

### Common Issues

**Database Connection Problems**
```bash
# Test connection
python test_database_connection.py

# Check PostgreSQL service
brew services start postgresql@17  # macOS
sudo systemctl start postgresql    # Linux
```

**pgvector Extension Issues**
```bash
# Install pgvector extension
psql -d argo_sih -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify installation
python scripts/validate_vector_schema.py
```

**Python Dependencies**
```bash
# Reinstall dependencies
pip install --upgrade -r backend/requirements.txt

# Check virtual environment
source .venv/bin/activate
which python  # Should point to .venv/bin/python
```

**Environment Variables**
- Ensure `.env` file exists with correct database credentials
- Check that `DATABASE_NAME=argo_sih` matches your PostgreSQL database
- Verify PostgreSQL user has necessary permissions

## 📈 Performance Considerations

- **Batch Processing**: Process NetCDF files in configurable batches
- **Vector Indexing**: Uses IVFFlat index for efficient similarity search
- **Memory Management**: Streaming processing for large datasets
- **Connection Pooling**: Optimized database connections
- **Caching**: Embedding and query result caching

## 🤝 Contributing

This project is part of Smart India Hackathon 2024. For development:

1. Follow the existing code structure and patterns
2. Add tests for new functionality
3. Update documentation for new features
4. Ensure compatibility with both local and cloud AI services

## 📄 License

This project is developed for Smart India Hackathon 2024. Please refer to the competition guidelines for usage and distribution terms.

## 🙏 Acknowledgments

- **ARGO Program**: For providing open oceanographic data
- **Smart India Hackathon**: For the opportunity to develop this solution
- **Open Source Community**: For the tools and libraries that make this possible

---

**Note**: This is a prototype system developed for demonstration purposes. For production use, additional security, monitoring, and scalability features should be implemented.