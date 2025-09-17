-- Argo Data Schema Creation
-- Database: argo_sih
-- This schema is designed for Argo float oceanographic data

-- Connect to the database
\c argo_sih;

-- Drop table if exists (for clean setup)
DROP TABLE IF EXISTS argo_profiles CASCADE;

-- Create main table for Argo profile data
CREATE TABLE argo_profiles (
    id SERIAL PRIMARY KEY,
    platform_number VARCHAR(20) NOT NULL,
    cycle_number INTEGER NOT NULL,
    profile_date TIMESTAMP,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    position_qc CHAR(1),
    level_index INTEGER NOT NULL,
    pressure REAL,
    temperature REAL,
    salinity REAL,
    pressure_qc CHAR(1),
    temperature_qc CHAR(1),
    salinity_qc CHAR(1),
    pressure_adjusted REAL,
    temperature_adjusted REAL,
    salinity_adjusted REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_platform_cycle ON argo_profiles(platform_number, cycle_number);
CREATE INDEX idx_location ON argo_profiles(latitude, longitude);
CREATE INDEX idx_date ON argo_profiles(profile_date);
CREATE INDEX idx_pressure ON argo_profiles(pressure);
CREATE INDEX idx_temperature ON argo_profiles(temperature);
CREATE INDEX idx_salinity ON argo_profiles(salinity);

-- Create a spatial index for geographic queries (if PostGIS is available)
-- CREATE INDEX idx_location_spatial ON argo_profiles USING GIST(ST_Point(longitude, latitude));

-- Create a table for file upload tracking
CREATE TABLE upload_history (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_size BIGINT,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'pending',
    records_inserted INTEGER DEFAULT 0,
    error_message TEXT
);

-- Create a table for data quality statistics
CREATE TABLE data_quality_stats (
    id SERIAL PRIMARY KEY,
    platform_number VARCHAR(20),
    total_profiles INTEGER,
    valid_temperature_count INTEGER,
    valid_salinity_count INTEGER,
    valid_pressure_count INTEGER,
    date_range_start TIMESTAMP,
    date_range_end TIMESTAMP,
    lat_min REAL,
    lat_max REAL,
    lon_min REAL,
    lon_max REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Display table structure
\d argo_profiles;
\d upload_history;
\d data_quality_stats;

-- Show current tables
\dt;

-- Sample query to verify setup
SELECT 'Database schema created successfully!' as status;