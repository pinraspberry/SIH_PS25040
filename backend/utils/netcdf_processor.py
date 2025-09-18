import netCDF4 as nc
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class ProfileData:
    """Data class for profile information"""
    platform_number: str
    cycle_number: int
    profile_date: Optional[datetime]
    latitude: float
    longitude: float
    measurements: List[Dict[str, Any]]
    quality_flags: Dict[str, Any]
    metadata: Dict[str, Any]

class ArgoNetCDFProcessor:
    """Enhanced NetCDF processor using xarray for robust Argo data handling"""
    
    # Variable name mappings for different Argo conventions
    VARIABLE_MAPPINGS = {
        'platform_number': ['PLATFORM_NUMBER', 'WMO_INST_TYPE'],
        'cycle_number': ['CYCLE_NUMBER', 'PROFILE_NUMBER'],
        'latitude': ['LATITUDE', 'LAT', 'JULD_LOCATION'],
        'longitude': ['LONGITUDE', 'LON', 'JULD_LOCATION'],
        'date': ['JULD', 'TIME', 'REFERENCE_DATE_TIME'],
        'pressure': ['PRES', 'PRESSURE', 'PRES_ADJUSTED'],
        'temperature': ['TEMP', 'TEMPERATURE', 'TEMP_ADJUSTED'],
        'salinity': ['PSAL', 'SALINITY', 'PSAL_ADJUSTED'],
        'pressure_qc': ['PRES_QC', 'PRESSURE_QC'],
        'temperature_qc': ['TEMP_QC', 'TEMPERATURE_QC'],
        'salinity_qc': ['PSAL_QC', 'SALINITY_QC']
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_dataset(self, file_path: str) -> xr.Dataset:
        """Load NetCDF file using xarray with error handling"""
        try:
            dataset = xr.open_dataset(file_path, decode_times=False)
            self.logger.info(f"Successfully loaded dataset from {file_path}")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset from {file_path}: {str(e)}")
            raise ValueError(f"Cannot load NetCDF file: {str(e)}")
    
    def print_dataset_summary(self, dataset: xr.Dataset) -> None:
        """Print dataset summary for debugging variable names"""
        print("\n=== DATASET SUMMARY ===")
        print(f"Dimensions: {dict(dataset.dims)}")
        print(f"Variables: {list(dataset.variables.keys())}")
        print(f"Attributes: {list(dataset.attrs.keys())}")
        
        print("\n=== VARIABLE DETAILS ===")
        for var_name in dataset.variables:
            var = dataset[var_name]
            print(f"{var_name}: shape={var.shape}, dtype={var.dtype}")
            if hasattr(var, 'long_name'):
                print(f"  long_name: {var.long_name}")
            if hasattr(var, 'units'):
                print(f"  units: {var.units}")
        print("========================\n")
    
    def find_variable(self, dataset: xr.Dataset, var_type: str) -> Optional[str]:
        """Find variable name using mapping conventions"""
        if var_type not in self.VARIABLE_MAPPINGS:
            return None
            
        for candidate in self.VARIABLE_MAPPINGS[var_type]:
            if candidate in dataset.variables:
                return candidate
        return None
    
    def validate_dataset(self, dataset: xr.Dataset) -> Tuple[bool, str]:
        """Validate dataset has required variables"""
        required_vars = ['platform_number', 'latitude', 'longitude', 'pressure']
        missing_vars = []
        
        for var_type in required_vars:
            if not self.find_variable(dataset, var_type):
                missing_vars.append(var_type)
        
        if missing_vars:
            self.print_dataset_summary(dataset)
            return False, f"Missing required variables: {missing_vars}"
        
        return True, "Dataset validation successful"
    
    def extract_scalar_value(self, dataset: xr.Dataset, var_name: str) -> Any:
        """Extract scalar value from dataset variable"""
        if var_name not in dataset.variables:
            return None
            
        data = dataset[var_name].values
        
        # Handle different data shapes
        if np.isscalar(data):
            return data
        elif hasattr(data, '__len__') and len(data) > 0:
            # Take first non-NaN value
            if isinstance(data, np.ndarray):
                valid_data = data[~np.isnan(data.astype(float))] if data.dtype.kind in 'fc' else data
                return valid_data[0] if len(valid_data) > 0 else None
            return data[0]
        return None
    
    def extract_array_data(self, dataset: xr.Dataset, var_name: str) -> np.ndarray:
        """Extract array data from dataset variable"""
        if var_name not in dataset.variables:
            return np.array([])
            
        data = dataset[var_name].values
        
        # Handle masked arrays
        if hasattr(data, 'mask'):
            data = np.ma.filled(data, np.nan)
        
        return np.asarray(data)
    
    def convert_argo_date(self, juld_value: Any) -> Optional[datetime]:
        """Convert Argo Julian date to datetime"""
        if juld_value is None or (isinstance(juld_value, float) and np.isnan(juld_value)):
            return None
            
        try:
            # Argo reference date is 1950-01-01
            reference_date = datetime(1950, 1, 1)
            return reference_date + pd.Timedelta(days=float(juld_value))
        except (ValueError, TypeError, OverflowError) as e:
            self.logger.warning(f"Failed to convert date {juld_value}: {str(e)}")
            return None
    
    def validate_measurement_data(self, pressure: np.ndarray, temperature: np.ndarray, 
                                salinity: np.ndarray) -> Dict[str, Any]:
        """Validate measurement data quality"""
        validation_results = {
            'valid_pressure_count': 0,
            'valid_temperature_count': 0,
            'valid_salinity_count': 0,
            'pressure_range': None,
            'temperature_range': None,
            'salinity_range': None,
            'quality_issues': []
        }
        
        # Validate pressure data
        if len(pressure) > 0:
            valid_pressure = pressure[~np.isnan(pressure)]
            validation_results['valid_pressure_count'] = len(valid_pressure)
            if len(valid_pressure) > 0:
                validation_results['pressure_range'] = (float(np.min(valid_pressure)), 
                                                      float(np.max(valid_pressure)))
                # Check for reasonable pressure values (0-6000 dbar)
                if np.any(valid_pressure < 0) or np.any(valid_pressure > 6000):
                    validation_results['quality_issues'].append("Pressure values outside expected range (0-6000 dbar)")
        
        # Validate temperature data
        if len(temperature) > 0:
            valid_temperature = temperature[~np.isnan(temperature)]
            validation_results['valid_temperature_count'] = len(valid_temperature)
            if len(valid_temperature) > 0:
                validation_results['temperature_range'] = (float(np.min(valid_temperature)), 
                                                         float(np.max(valid_temperature)))
                # Check for reasonable temperature values (-5 to 40°C)
                if np.any(valid_temperature < -5) or np.any(valid_temperature > 40):
                    validation_results['quality_issues'].append("Temperature values outside expected range (-5 to 40°C)")
        
        # Validate salinity data
        if len(salinity) > 0:
            valid_salinity = salinity[~np.isnan(salinity)]
            validation_results['valid_salinity_count'] = len(valid_salinity)
            if len(valid_salinity) > 0:
                validation_results['salinity_range'] = (float(np.min(valid_salinity)), 
                                                       float(np.max(valid_salinity)))
                # Check for reasonable salinity values (0-50 PSU)
                if np.any(valid_salinity < 0) or np.any(valid_salinity > 50):
                    validation_results['quality_issues'].append("Salinity values outside expected range (0-50 PSU)")
        
        return validation_results
    
    def extract_profiles(self, dataset: xr.Dataset) -> List[ProfileData]:
        """Extract all profiles from dataset"""
        profiles = []
        
        try:
            # Find variable names using mappings
            platform_var = self.find_variable(dataset, 'platform_number')
            cycle_var = self.find_variable(dataset, 'cycle_number')
            lat_var = self.find_variable(dataset, 'latitude')
            lon_var = self.find_variable(dataset, 'longitude')
            date_var = self.find_variable(dataset, 'date')
            pres_var = self.find_variable(dataset, 'pressure')
            temp_var = self.find_variable(dataset, 'temperature')
            sal_var = self.find_variable(dataset, 'salinity')
            
            # Extract basic metadata
            platform_raw = self.extract_scalar_value(dataset, platform_var) if platform_var else "UNKNOWN"
            platform_number = str(platform_raw).strip().replace("b'", "").replace("'", "") if platform_raw else "UNKNOWN"
            cycle_number = int(self.extract_scalar_value(dataset, cycle_var)) if cycle_var else 0
            latitude = float(self.extract_scalar_value(dataset, lat_var)) if lat_var else 0.0
            longitude = float(self.extract_scalar_value(dataset, lon_var)) if lon_var else 0.0
            
            # Extract and convert date
            juld_value = self.extract_scalar_value(dataset, date_var) if date_var else None
            profile_date = self.convert_argo_date(juld_value)
            
            # Extract measurement arrays
            pressure = self.extract_array_data(dataset, pres_var) if pres_var else np.array([])
            temperature = self.extract_array_data(dataset, temp_var) if temp_var else np.array([])
            salinity = self.extract_array_data(dataset, sal_var) if sal_var else np.array([])
            
            # Extract QC flags
            pres_qc_var = self.find_variable(dataset, 'pressure_qc')
            temp_qc_var = self.find_variable(dataset, 'temperature_qc')
            sal_qc_var = self.find_variable(dataset, 'salinity_qc')
            
            pressure_qc = self.extract_array_data(dataset, pres_qc_var) if pres_qc_var else np.array(['0'] * len(pressure))
            temperature_qc = self.extract_array_data(dataset, temp_qc_var) if temp_qc_var else np.array(['0'] * len(temperature))
            salinity_qc = self.extract_array_data(dataset, sal_qc_var) if sal_qc_var else np.array(['0'] * len(salinity))
            
            # Validate data quality
            validation_results = self.validate_measurement_data(pressure, temperature, salinity)
            
            # Create measurements list
            measurements = []
            max_levels = max(len(pressure), len(temperature), len(salinity))
            
            for i in range(max_levels):
                # Handle multi-dimensional arrays by flattening
                def safe_extract_value(arr, idx):
                    if len(arr) == 0:
                        return None
                    if arr.ndim > 1:
                        # Flatten multi-dimensional arrays
                        flat_arr = arr.flatten()
                        if idx < len(flat_arr):
                            val = flat_arr[idx]
                            try:
                                return None if np.isnan(float(val)) else float(val)
                            except (ValueError, TypeError):
                                return None
                        return None
                    else:
                        if idx < len(arr):
                            val = arr[idx]
                            try:
                                return None if np.isnan(float(val)) else float(val)
                            except (ValueError, TypeError):
                                return None
                        return None
                
                def safe_extract_qc(arr, idx):
                    if len(arr) == 0:
                        return '0'
                    if arr.ndim > 1:
                        flat_arr = arr.flatten()
                        if idx < len(flat_arr):
                            qc_val = str(flat_arr[idx]).strip()
                            # Handle bytes strings and ensure single character
                            qc_val = qc_val.replace("b'", "").replace("'", "")
                            return qc_val[0] if qc_val else '0'
                        return '0'
                    else:
                        if idx < len(arr):
                            qc_val = str(arr[idx]).strip()
                            # Handle bytes strings and ensure single character
                            qc_val = qc_val.replace("b'", "").replace("'", "")
                            return qc_val[0] if qc_val else '0'
                        return '0'
                
                measurement = {
                    'level_index': i,
                    'pressure': safe_extract_value(pressure, i),
                    'temperature': safe_extract_value(temperature, i),
                    'salinity': safe_extract_value(salinity, i),
                    'pressure_qc': safe_extract_qc(pressure_qc, i),
                    'temperature_qc': safe_extract_qc(temperature_qc, i),
                    'salinity_qc': safe_extract_qc(salinity_qc, i)
                }
                
                # Only add measurements with at least one valid value
                if any(measurement[key] is not None for key in ['pressure', 'temperature', 'salinity']):
                    measurements.append(measurement)
            
            # Create profile data
            profile = ProfileData(
                platform_number=platform_number,
                cycle_number=cycle_number,
                profile_date=profile_date,
                latitude=latitude,
                longitude=longitude,
                measurements=measurements,
                quality_flags={
                    'pressure_qc': pressure_qc.tolist() if len(pressure_qc) > 0 else [],
                    'temperature_qc': temperature_qc.tolist() if len(temperature_qc) > 0 else [],
                    'salinity_qc': salinity_qc.tolist() if len(salinity_qc) > 0 else []
                },
                metadata={
                    'validation_results': validation_results,
                    'total_levels': len(measurements),
                    'dataset_attrs': dict(dataset.attrs) if hasattr(dataset, 'attrs') else {}
                }
            )
            
            profiles.append(profile)
            self.logger.info(f"Extracted profile: Platform {platform_number}, Cycle {cycle_number}, {len(measurements)} levels")
            
        except Exception as e:
            self.logger.error(f"Error extracting profiles: {str(e)}")
            raise ValueError(f"Profile extraction failed: {str(e)}")
        
        return profiles

class ProfileSummarizer:
    """Generate structured text summaries from profile data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_summary(self, profile: ProfileData) -> str:
        """Generate a structured text summary of the profile"""
        try:
            # Basic location and metadata
            location_desc = self._get_location_description(profile.latitude, profile.longitude)
            date_str = profile.profile_date.strftime("%Y-%m-%d %H:%M UTC") if profile.profile_date else "Unknown date"
            
            # Measurement statistics
            temp_stats = self._calculate_measurement_stats(profile.measurements, 'temperature')
            sal_stats = self._calculate_measurement_stats(profile.measurements, 'salinity')
            pres_stats = self._calculate_measurement_stats(profile.measurements, 'pressure')
            
            # Quality assessment
            quality_summary = self._assess_data_quality(profile)
            
            # Generate summary text
            summary_parts = [
                f"Location: {profile.latitude:.3f}°N, {profile.longitude:.3f}°E ({location_desc})",
                f"Date: {date_str}",
                f"Platform: {profile.platform_number}, Cycle: {profile.cycle_number}",
                f"Depth Range: {pres_stats['min']:.1f} - {pres_stats['max']:.1f} dbar ({pres_stats['count']} levels)" if pres_stats['count'] > 0 else "Depth: No valid pressure data",
                f"Temperature: {temp_stats['min']:.2f} - {temp_stats['max']:.2f}°C (avg: {temp_stats['avg']:.2f}°C)" if temp_stats['count'] > 0 else "Temperature: No valid data",
                f"Salinity: {sal_stats['min']:.2f} - {sal_stats['max']:.2f} PSU (avg: {sal_stats['avg']:.2f} PSU)" if sal_stats['count'] > 0 else "Salinity: No valid data",
                f"Data Quality: {quality_summary}"
            ]
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating profile summary: {str(e)}")
            return f"Profile summary error: Platform {profile.platform_number}, Cycle {profile.cycle_number}"
    
    def _get_location_description(self, lat: float, lon: float) -> str:
        """Get a descriptive location name based on coordinates"""
        # Simple ocean basin classification
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            if lat > 66:  # Arctic Circle
                return "Arctic Ocean"
            elif lat < -60:  # Southern Ocean
                return "Southern Ocean"
            elif -20 <= lat <= 30 and 30 <= lon <= 120:
                return "Indian Ocean"
            elif -60 <= lat <= 70 and -100 <= lon <= 20:
                return "Atlantic Ocean"
            elif -60 <= lat <= 60 and 100 <= lon <= 180:
                return "Pacific Ocean"
            else:
                return "Global Ocean"
        return "Unknown location"
    
    def _calculate_measurement_stats(self, measurements: List[Dict], param: str) -> Dict[str, float]:
        """Calculate statistics for a measurement parameter"""
        values = [m[param] for m in measurements if m[param] is not None]
        
        if not values:
            return {'count': 0, 'min': 0, 'max': 0, 'avg': 0}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values)
        }
    
    def _assess_data_quality(self, profile: ProfileData) -> str:
        """Assess overall data quality"""
        validation = profile.metadata.get('validation_results', {})
        quality_issues = validation.get('quality_issues', [])
        
        total_measurements = len(profile.measurements)
        valid_temp = validation.get('valid_temperature_count', 0)
        valid_sal = validation.get('valid_salinity_count', 0)
        valid_pres = validation.get('valid_pressure_count', 0)
        
        if quality_issues:
            return f"Issues detected: {'; '.join(quality_issues[:2])}"
        elif total_measurements == 0:
            return "No valid measurements"
        elif valid_temp == 0 and valid_sal == 0:
            return "Missing T/S data"
        elif valid_temp / total_measurements > 0.8 and valid_sal / total_measurements > 0.8:
            return "Good quality"
        else:
            return "Partial data"

class NetCDFProcessor:
    """Legacy NetCDF processor - maintained for backward compatibility"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.enhanced_processor = ArgoNetCDFProcessor()
        self.summarizer = ProfileSummarizer()
    
    def validate_file(self, file_path):
        """Validate NetCDF file format and required variables"""
        try:
            dataset = self.enhanced_processor.load_dataset(file_path)
            is_valid, message = self.enhanced_processor.validate_dataset(dataset)
            dataset.close()
            return is_valid, message
        except Exception as e:
            return False, f"File validation error: {str(e)}"
    
    def extract_profile_data(self, file_path):
        """Extract profile data from NetCDF file using enhanced processor"""
        try:
            dataset = self.enhanced_processor.load_dataset(file_path)
            profiles = self.enhanced_processor.extract_profiles(dataset)
            dataset.close()
            
            # Convert to legacy format for backward compatibility
            records = []
            for profile in profiles:
                for measurement in profile.measurements:
                    # Clean platform number from bytes strings
                    clean_platform = str(profile.platform_number).strip().replace("b'", "").replace("'", "")
                    
                    record = {
                        'platform_number': clean_platform,
                        'cycle_number': profile.cycle_number,
                        'profile_date': profile.profile_date,
                        'latitude': profile.latitude,
                        'longitude': profile.longitude,
                        'level_index': measurement['level_index'],
                        'pressure': measurement['pressure'],
                        'temperature': measurement['temperature'],
                        'salinity': measurement['salinity'],
                        'pressure_qc': measurement['pressure_qc'][:1] if measurement['pressure_qc'] else '0',
                        'temperature_qc': measurement['temperature_qc'][:1] if measurement['temperature_qc'] else '0',
                        'salinity_qc': measurement['salinity_qc'][:1] if measurement['salinity_qc'] else '0',
                        'pressure_adjusted': None,  # Legacy compatibility
                        'temperature_adjusted': None,
                        'salinity_adjusted': None
                    }
                    records.append(record)
            
            return records
                
        except Exception as e:
            self.logger.error(f"Error extracting profile data: {str(e)}")
            raise
    
    def extract_profiles_enhanced(self, file_path):
        """Extract profiles using enhanced processor with full metadata"""
        try:
            dataset = self.enhanced_processor.load_dataset(file_path)
            profiles = self.enhanced_processor.extract_profiles(dataset)
            dataset.close()
            return profiles
        except Exception as e:
            self.logger.error(f"Error extracting enhanced profiles: {str(e)}")
            raise
    
    def generate_profile_summary(self, profile_data):
        """Generate text summary for a profile"""
        if isinstance(profile_data, ProfileData):
            return self.summarizer.generate_summary(profile_data)
        else:
            # Handle legacy format
            self.logger.warning("Using legacy profile format for summary generation")
            return f"Profile: Platform {profile_data.get('platform_number', 'Unknown')}, Cycle {profile_data.get('cycle_number', 0)}"
    
    def _get_variable_data(self, dataset, var_name, default=None):
        """Safely extract variable data from dataset"""
        if var_name in dataset.variables:
            data = dataset.variables[var_name][:]
            if hasattr(data, 'mask'):
                data = np.ma.filled(data, np.nan)
            return data
        return default
    
    def _convert_julian_date(self, juld):
        """Convert Julian date to datetime"""
        if juld is None or np.isnan(juld):
            return None
        
        # Argo reference date is 1950-01-01
        reference_date = datetime(1950, 1, 1)
        try:
            return reference_date + pd.Timedelta(days=float(juld))
        except:
            return None
    
    def _format_profile_data(self, platform_number, cycle_number, profile_date,
                           latitude, longitude, pressure, temperature, salinity,
                           pressure_qc, temp_qc, psal_qc,
                           pressure_adj, temp_adj, psal_adj):
        """Format extracted data into records for database insertion"""
        
        records = []
        
        # Handle scalar vs array data
        if np.isscalar(latitude):
            lat_val = float(latitude) if not np.isnan(latitude) else None
            lon_val = float(longitude) if not np.isnan(longitude) else None
            
            # Handle pressure levels
            if hasattr(pressure, '__len__'):
                for i, pres_val in enumerate(pressure):
                    if not np.isnan(pres_val):
                        record = {
                            'platform_number': str(platform_number).strip() if platform_number else None,
                            'cycle_number': int(cycle_number) if cycle_number else 0,
                            'profile_date': profile_date,
                            'latitude': lat_val,
                            'longitude': lon_val,
                            'level_index': i,
                            'pressure': float(pres_val),
                            'temperature': float(temperature[i]) if i < len(temperature) and not np.isnan(temperature[i]) else None,
                            'salinity': float(salinity[i]) if i < len(salinity) and not np.isnan(salinity[i]) else None,
                            'pressure_qc': str(pressure_qc[i]) if hasattr(pressure_qc, '__len__') and i < len(pressure_qc) else '0',
                            'temperature_qc': str(temp_qc[i]) if hasattr(temp_qc, '__len__') and i < len(temp_qc) else '0',
                            'salinity_qc': str(psal_qc[i]) if hasattr(psal_qc, '__len__') and i < len(psal_qc) else '0',
                            'pressure_adjusted': float(pressure_adj[i]) if pressure_adj is not None and i < len(pressure_adj) and not np.isnan(pressure_adj[i]) else None,
                            'temperature_adjusted': float(temp_adj[i]) if temp_adj is not None and i < len(temp_adj) and not np.isnan(temp_adj[i]) else None,
                            'salinity_adjusted': float(psal_adj[i]) if psal_adj is not None and i < len(psal_adj) and not np.isnan(psal_adj[i]) else None
                        }
                        records.append(record)
        
        return records