import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
import logging

class NetCDFProcessor:
    """Process NetCDF Argo float data files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_file(self, file_path):
        """Validate NetCDF file format and required variables"""
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                required_vars = ['PRES', 'TEMP', 'PSAL', 'LATITUDE', 'LONGITUDE']
                missing_vars = [var for var in required_vars if var not in dataset.variables]
                
                if missing_vars:
                    return False, f"Missing required variables: {missing_vars}"
                
                return True, "File validation successful"
        except Exception as e:
            return False, f"File validation error: {str(e)}"
    
    def extract_profile_data(self, file_path):
        """Extract profile data from NetCDF file"""
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                # Extract basic metadata
                platform_number = self._get_variable_data(dataset, 'PLATFORM_NUMBER')
                cycle_number = self._get_variable_data(dataset, 'CYCLE_NUMBER')
                
                # Extract coordinates
                latitude = self._get_variable_data(dataset, 'LATITUDE')
                longitude = self._get_variable_data(dataset, 'LONGITUDE')
                
                # Extract date
                juld = self._get_variable_data(dataset, 'JULD')
                profile_date = self._convert_julian_date(juld)
                
                # Extract measurements
                pressure = self._get_variable_data(dataset, 'PRES')
                temperature = self._get_variable_data(dataset, 'TEMP')
                salinity = self._get_variable_data(dataset, 'PSAL')
                
                # Extract QC flags
                pressure_qc = self._get_variable_data(dataset, 'PRES_QC', default='0')
                temp_qc = self._get_variable_data(dataset, 'TEMP_QC', default='0')
                psal_qc = self._get_variable_data(dataset, 'PSAL_QC', default='0')
                
                # Extract adjusted values if available
                pressure_adj = self._get_variable_data(dataset, 'PRES_ADJUSTED', default=None)
                temp_adj = self._get_variable_data(dataset, 'TEMP_ADJUSTED', default=None)
                psal_adj = self._get_variable_data(dataset, 'PSAL_ADJUSTED', default=None)
                
                return self._format_profile_data(
                    platform_number, cycle_number, profile_date,
                    latitude, longitude, pressure, temperature, salinity,
                    pressure_qc, temp_qc, psal_qc,
                    pressure_adj, temp_adj, psal_adj
                )
                
        except Exception as e:
            self.logger.error(f"Error extracting profile data: {str(e)}")
            raise
    
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