import unittest
import numpy as np
import xarray as xr
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.utils.netcdf_processor import ArgoNetCDFProcessor, ProfileSummarizer, ProfileData, NetCDFProcessor


class TestArgoNetCDFProcessor(unittest.TestCase):
    """Test cases for ArgoNetCDFProcessor"""
    
    def setUp(self):
        self.processor = ArgoNetCDFProcessor()
        self.summarizer = ProfileSummarizer()
    
    def create_test_dataset(self, include_all_vars=True, corrupt_data=False):
        """Create a test xarray dataset mimicking Argo data"""
        n_levels = 10
        
        # Create test data
        data_vars = {}
        
        if include_all_vars:
            data_vars.update({
                'PLATFORM_NUMBER': (['N_PROF'], ['1901234']),
                'CYCLE_NUMBER': (['N_PROF'], [1]),
                'LATITUDE': (['N_PROF'], [25.5]),
                'LONGITUDE': (['N_PROF'], [65.2]),
                'JULD': (['N_PROF'], [25567.5]),  # Days since 1950-01-01
                'PRES': (['N_PROF', 'N_LEVELS'], [np.linspace(5, 2000, n_levels)]),
                'TEMP': (['N_PROF', 'N_LEVELS'], [np.linspace(28, 2, n_levels)]),
                'PSAL': (['N_PROF', 'N_LEVELS'], [np.linspace(35, 34.5, n_levels)]),
                'PRES_QC': (['N_PROF', 'N_LEVELS'], [['1'] * n_levels]),
                'TEMP_QC': (['N_PROF', 'N_LEVELS'], [['1'] * n_levels]),
                'PSAL_QC': (['N_PROF', 'N_LEVELS'], [['1'] * n_levels])
            })
        else:
            # Missing some required variables (missing LATITUDE which is required)
            data_vars.update({
                'PLATFORM_NUMBER': (['N_PROF'], ['1901234']),
                'LONGITUDE': (['N_PROF'], [65.2]),
                'PRES': (['N_PROF', 'N_LEVELS'], [np.linspace(5, 2000, n_levels)])
            })
        
        if corrupt_data:
            # Add some corrupted data
            temp_data = np.linspace(28, 2, n_levels)
            temp_data[3:6] = np.nan  # Some NaN values
            temp_data[7] = 999  # Unrealistic value
            data_vars['TEMP'] = (['N_PROF', 'N_LEVELS'], [temp_data])
        
        coords = {
            'N_PROF': [0],
            'N_LEVELS': list(range(n_levels))
        }
        
        attrs = {
            'title': 'Test Argo Profile',
            'institution': 'Test Institution',
            'Conventions': 'Argo-3.1'
        }
        
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)
    
    def test_load_dataset_success(self):
        """Test successful dataset loading"""
        # Create temporary NetCDF file
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
            dataset = self.create_test_dataset()
            dataset.to_netcdf(tmp_file.name)
            tmp_file.flush()
            
            # Test loading
            loaded_dataset = self.processor.load_dataset(tmp_file.name)
            self.assertIsInstance(loaded_dataset, xr.Dataset)
            self.assertIn('PLATFORM_NUMBER', loaded_dataset.variables)
            
            loaded_dataset.close()
            os.unlink(tmp_file.name)
    
    def test_load_dataset_failure(self):
        """Test dataset loading failure"""
        with self.assertRaises(ValueError):
            self.processor.load_dataset('nonexistent_file.nc')
    
    def test_find_variable_success(self):
        """Test variable finding with mapping"""
        dataset = self.create_test_dataset()
        
        # Test finding existing variables
        self.assertEqual(self.processor.find_variable(dataset, 'platform_number'), 'PLATFORM_NUMBER')
        self.assertEqual(self.processor.find_variable(dataset, 'latitude'), 'LATITUDE')
        self.assertEqual(self.processor.find_variable(dataset, 'pressure'), 'PRES')
        
        # Test finding non-existent variable type
        self.assertIsNone(self.processor.find_variable(dataset, 'nonexistent_type'))
    
    def test_validate_dataset_success(self):
        """Test successful dataset validation"""
        dataset = self.create_test_dataset()
        is_valid, message = self.processor.validate_dataset(dataset)
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Dataset validation successful")
    
    def test_validate_dataset_failure(self):
        """Test dataset validation failure"""
        dataset = self.create_test_dataset(include_all_vars=False)
        is_valid, message = self.processor.validate_dataset(dataset)
        
        self.assertFalse(is_valid)
        self.assertIn("Missing required variables", message)
    
    def test_extract_scalar_value(self):
        """Test scalar value extraction"""
        dataset = self.create_test_dataset()
        
        platform = self.processor.extract_scalar_value(dataset, 'PLATFORM_NUMBER')
        self.assertEqual(platform, '1901234')
        
        latitude = self.processor.extract_scalar_value(dataset, 'LATITUDE')
        self.assertEqual(latitude, 25.5)
        
        # Test non-existent variable
        result = self.processor.extract_scalar_value(dataset, 'NONEXISTENT')
        self.assertIsNone(result)
    
    def test_extract_array_data(self):
        """Test array data extraction"""
        dataset = self.create_test_dataset()
        
        pressure = self.processor.extract_array_data(dataset, 'PRES')
        self.assertIsInstance(pressure, np.ndarray)
        self.assertEqual(len(pressure.shape), 2)  # Should be 2D array
        
        # Test non-existent variable
        result = self.processor.extract_array_data(dataset, 'NONEXISTENT')
        self.assertEqual(len(result), 0)
    
    def test_convert_argo_date(self):
        """Test Argo date conversion"""
        # Test valid date
        juld = 25567.5  # Should be around 2020
        result = self.processor.convert_argo_date(juld)
        self.assertIsInstance(result, datetime)
        self.assertEqual(result.year, 2020)
        
        # Test invalid dates
        self.assertIsNone(self.processor.convert_argo_date(None))
        self.assertIsNone(self.processor.convert_argo_date(np.nan))
        self.assertIsNone(self.processor.convert_argo_date('invalid'))
    
    def test_validate_measurement_data(self):
        """Test measurement data validation"""
        # Create test data
        pressure = np.array([5, 10, 20, 50, 100])
        temperature = np.array([28, 25, 20, 15, 10])
        salinity = np.array([35.0, 35.1, 35.2, 35.3, 35.4])
        
        validation = self.processor.validate_measurement_data(pressure, temperature, salinity)
        
        self.assertEqual(validation['valid_pressure_count'], 5)
        self.assertEqual(validation['valid_temperature_count'], 5)
        self.assertEqual(validation['valid_salinity_count'], 5)
        self.assertEqual(validation['pressure_range'], (5.0, 100.0))
        self.assertEqual(validation['temperature_range'], (10.0, 28.0))
        self.assertEqual(validation['salinity_range'], (35.0, 35.4))
        self.assertEqual(len(validation['quality_issues']), 0)
    
    def test_validate_measurement_data_with_issues(self):
        """Test measurement data validation with quality issues"""
        # Create data with issues
        pressure = np.array([5, 10, -5, 7000])  # Negative and too high pressure
        temperature = np.array([28, 25, -10, 50])  # Too low and too high temperature
        salinity = np.array([35.0, 35.1, -1, 60])  # Negative and too high salinity
        
        validation = self.processor.validate_measurement_data(pressure, temperature, salinity)
        
        self.assertGreater(len(validation['quality_issues']), 0)
        self.assertTrue(any('Pressure' in issue for issue in validation['quality_issues']))
        self.assertTrue(any('Temperature' in issue for issue in validation['quality_issues']))
        self.assertTrue(any('Salinity' in issue for issue in validation['quality_issues']))
    
    def test_extract_profiles_success(self):
        """Test successful profile extraction"""
        dataset = self.create_test_dataset()
        profiles = self.processor.extract_profiles(dataset)
        
        self.assertEqual(len(profiles), 1)
        profile = profiles[0]
        
        self.assertIsInstance(profile, ProfileData)
        self.assertEqual(profile.platform_number, '1901234')
        self.assertEqual(profile.cycle_number, 1)
        self.assertEqual(profile.latitude, 25.5)
        self.assertEqual(profile.longitude, 65.2)
        self.assertIsInstance(profile.profile_date, datetime)
        self.assertGreater(len(profile.measurements), 0)
        
        # Check measurements structure
        measurement = profile.measurements[0]
        self.assertIn('level_index', measurement)
        self.assertIn('pressure', measurement)
        self.assertIn('temperature', measurement)
        self.assertIn('salinity', measurement)
        self.assertIn('pressure_qc', measurement)
    
    def test_extract_profiles_with_missing_data(self):
        """Test profile extraction with missing data"""
        dataset = self.create_test_dataset(include_all_vars=False)
        
        # Should handle missing variables gracefully
        with patch.object(self.processor, 'print_dataset_summary'):
            profiles = self.processor.extract_profiles(dataset)
            
        self.assertEqual(len(profiles), 1)
        profile = profiles[0]
        self.assertEqual(profile.platform_number, '1901234')
        # Should have default values for missing data
        self.assertEqual(profile.cycle_number, 0)


class TestProfileSummarizer(unittest.TestCase):
    """Test cases for ProfileSummarizer"""
    
    def setUp(self):
        self.summarizer = ProfileSummarizer()
    
    def create_test_profile(self):
        """Create a test ProfileData object"""
        measurements = [
            {
                'level_index': 0,
                'pressure': 5.0,
                'temperature': 28.0,
                'salinity': 35.0,
                'pressure_qc': '1',
                'temperature_qc': '1',
                'salinity_qc': '1'
            },
            {
                'level_index': 1,
                'pressure': 100.0,
                'temperature': 15.0,
                'salinity': 35.2,
                'pressure_qc': '1',
                'temperature_qc': '1',
                'salinity_qc': '1'
            }
        ]
        
        return ProfileData(
            platform_number='1901234',
            cycle_number=1,
            profile_date=datetime(2020, 6, 15, 12, 0),
            latitude=25.5,
            longitude=65.2,
            measurements=measurements,
            quality_flags={
                'pressure_qc': ['1', '1'],
                'temperature_qc': ['1', '1'],
                'salinity_qc': ['1', '1']
            },
            metadata={
                'validation_results': {
                    'valid_pressure_count': 2,
                    'valid_temperature_count': 2,
                    'valid_salinity_count': 2,
                    'quality_issues': []
                }
            }
        )
    
    def test_generate_summary(self):
        """Test summary generation"""
        profile = self.create_test_profile()
        summary = self.summarizer.generate_summary(profile)
        
        self.assertIsInstance(summary, str)
        self.assertIn('25.500°N, 65.200°E', summary)
        self.assertIn('Indian Ocean', summary)
        self.assertIn('2020-06-15 12:00 UTC', summary)
        self.assertIn('Platform: 1901234', summary)
        self.assertIn('Cycle: 1', summary)
        self.assertIn('5.0 - 100.0 dbar', summary)
        self.assertIn('15.00 - 28.00°C', summary)
        self.assertIn('35.00 - 35.20 PSU', summary)
    
    def test_get_location_description(self):
        """Test location description generation"""
        # Test Indian Ocean
        desc = self.summarizer._get_location_description(25.5, 65.2)
        self.assertEqual(desc, "Indian Ocean")
        
        # Test Atlantic Ocean
        desc = self.summarizer._get_location_description(40.0, -30.0)
        self.assertEqual(desc, "Atlantic Ocean")
        
        # Test Pacific Ocean
        desc = self.summarizer._get_location_description(30.0, 150.0)
        self.assertEqual(desc, "Pacific Ocean")
        
        # Test Arctic Ocean
        desc = self.summarizer._get_location_description(70.0, 0.0)
        self.assertEqual(desc, "Arctic Ocean")
        
        # Test Southern Ocean
        desc = self.summarizer._get_location_description(-70.0, 0.0)
        self.assertEqual(desc, "Southern Ocean")
    
    def test_calculate_measurement_stats(self):
        """Test measurement statistics calculation"""
        measurements = [
            {'temperature': 28.0, 'salinity': 35.0, 'pressure': 5.0},
            {'temperature': 15.0, 'salinity': 35.2, 'pressure': 100.0},
            {'temperature': None, 'salinity': 35.1, 'pressure': 50.0}
        ]
        
        temp_stats = self.summarizer._calculate_measurement_stats(measurements, 'temperature')
        self.assertEqual(temp_stats['count'], 2)
        self.assertEqual(temp_stats['min'], 15.0)
        self.assertEqual(temp_stats['max'], 28.0)
        self.assertEqual(temp_stats['avg'], 21.5)
        
        # Test with no valid data
        empty_measurements = [{'temperature': None}, {'temperature': None}]
        temp_stats = self.summarizer._calculate_measurement_stats(empty_measurements, 'temperature')
        self.assertEqual(temp_stats['count'], 0)
    
    def test_assess_data_quality(self):
        """Test data quality assessment"""
        # Test good quality profile
        profile = self.create_test_profile()
        quality = self.summarizer._assess_data_quality(profile)
        self.assertEqual(quality, "Good quality")
        
        # Test profile with issues
        profile.metadata['validation_results']['quality_issues'] = ['Temperature values outside expected range']
        quality = self.summarizer._assess_data_quality(profile)
        self.assertIn("Issues detected", quality)
        
        # Test profile with no measurements
        profile.measurements = []
        profile.metadata['validation_results']['quality_issues'] = []
        quality = self.summarizer._assess_data_quality(profile)
        self.assertEqual(quality, "No valid measurements")


class TestNetCDFProcessorLegacy(unittest.TestCase):
    """Test cases for legacy NetCDFProcessor compatibility"""
    
    def setUp(self):
        self.processor = NetCDFProcessor()
    
    def create_test_netcdf_file(self):
        """Create a temporary NetCDF file for testing"""
        dataset = xr.Dataset({
            'PLATFORM_NUMBER': (['N_PROF'], ['1901234']),
            'CYCLE_NUMBER': (['N_PROF'], [1]),
            'LATITUDE': (['N_PROF'], [25.5]),
            'LONGITUDE': (['N_PROF'], [65.2]),
            'JULD': (['N_PROF'], [25567.5]),
            'PRES': (['N_PROF', 'N_LEVELS'], [np.linspace(5, 100, 5)]),
            'TEMP': (['N_PROF', 'N_LEVELS'], [np.linspace(28, 15, 5)]),
            'PSAL': (['N_PROF', 'N_LEVELS'], [np.linspace(35, 35.2, 5)])
        }, coords={'N_PROF': [0], 'N_LEVELS': list(range(5))})
        
        tmp_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        dataset.to_netcdf(tmp_file.name)
        tmp_file.close()
        return tmp_file.name
    
    def test_validate_file_legacy(self):
        """Test legacy validate_file method"""
        tmp_file = self.create_test_netcdf_file()
        
        try:
            is_valid, message = self.processor.validate_file(tmp_file)
            self.assertTrue(is_valid)
            self.assertEqual(message, "Dataset validation successful")
        finally:
            os.unlink(tmp_file)
    
    def test_extract_profile_data_legacy(self):
        """Test legacy extract_profile_data method"""
        tmp_file = self.create_test_netcdf_file()
        
        try:
            records = self.processor.extract_profile_data(tmp_file)
            self.assertIsInstance(records, list)
            self.assertGreater(len(records), 0)
            
            # Check record structure
            record = records[0]
            self.assertIn('platform_number', record)
            self.assertIn('cycle_number', record)
            self.assertIn('latitude', record)
            self.assertIn('longitude', record)
            self.assertIn('level_index', record)
            self.assertIn('pressure', record)
            self.assertIn('temperature', record)
            self.assertIn('salinity', record)
        finally:
            os.unlink(tmp_file)
    
    def test_extract_profiles_enhanced(self):
        """Test enhanced profile extraction method"""
        tmp_file = self.create_test_netcdf_file()
        
        try:
            profiles = self.processor.extract_profiles_enhanced(tmp_file)
            self.assertIsInstance(profiles, list)
            self.assertEqual(len(profiles), 1)
            
            profile = profiles[0]
            self.assertIsInstance(profile, ProfileData)
            self.assertEqual(profile.platform_number, '1901234')
        finally:
            os.unlink(tmp_file)
    
    def test_generate_profile_summary(self):
        """Test profile summary generation"""
        tmp_file = self.create_test_netcdf_file()
        
        try:
            profiles = self.processor.extract_profiles_enhanced(tmp_file)
            profile = profiles[0]
            
            summary = self.processor.generate_profile_summary(profile)
            self.assertIsInstance(summary, str)
            self.assertIn('1901234', summary)
            self.assertIn('Indian Ocean', summary)
        finally:
            os.unlink(tmp_file)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    unittest.main()