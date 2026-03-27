__author__ = "saeedamen"  # Saeed Amen

#
# Copyright 2016 Cuemacro
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on a "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import pandas as pd

from findatapy.market.ioengine import IOEngine

from findatapy.util.dataconstants import DataConstants

data_constants = DataConstants()

redis_server = data_constants.db_cache_server
redis_port = data_constants.db_cache_port

def test_redis_caching():
    # Note: you need to install Redis in order for this to work!

    # read CSV from disk, and make sure to parse dates
    df = pd.read_csv("S&P500.csv", parse_dates=['Date'], index_col=['Date'])
    df.index = pd.to_datetime(df.index)

    io = IOEngine()

    use_cache_compression = [True, False]

    for u in use_cache_compression:
        # Write DataFrame to Redis (using pyarrow format)
        io.write_time_series_cache_to_disk('test_key', df, engine='redis', db_server=redis_server, db_port=redis_port,
                                           use_cache_compression=u)

        # Read back DataFrame from Redis (using pyarrow format)
        df_out = io.read_time_series_cache_from_disk('test_key', engine='redis', db_server=redis_server, db_port=redis_port)

        pd.testing.assert_frame_equal(df, df_out)


def test_path_join():

    io = IOEngine()

    path = io.path_join("/home/hello", "hello", "hello")

    assert path == "/home/hello/hello/hello"

    path = io.path_join("s3://home/hello", "hello", "hello")

    assert path == "s3://home/hello/hello/hello"


def test_get_file_properties_and_touch():
    """Test get_file_properties, touch_file, and is_same_file methods"""
    import time
    import tempfile
    import os
    
    io = IOEngine()
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_file_1 = f.name
        f.write("Hello World")
    
    try:
        # Get initial file properties
        props_1 = io.get_file_properties(temp_file_1)
        
        assert props_1 is not None
        assert 'modified_datetime' in props_1
        assert 'filesize_bytes' in props_1
        assert 'path' in props_1
        assert props_1['filesize_bytes'] == 11  # "Hello World" is 11 bytes
        
        # Sleep briefly to ensure time difference
        time.sleep(0.1)
        
        # Touch the file (update modification time)
        success = io.touch_file(temp_file_1)
        assert success is True
        
        # Get properties again
        props_2 = io.get_file_properties(temp_file_1)
        
        # Modified time should be different after touch
        assert props_2['modified_datetime'] != props_1['modified_datetime']
        assert props_2['modified_datetime'] > props_1['modified_datetime']
        
        # But size should be the same
        assert props_2['filesize_bytes'] == props_1['filesize_bytes']
        
        # When comparing the same file path, it will return True based on path comparison
        # But if we strip out the path from metadata and compare just metadata, 
        # files should NOT be the same due to different modification times
        props_1_no_path = {'modified_datetime': props_1['modified_datetime'], 
                          'filesize_bytes': props_1['filesize_bytes']}
        props_2_no_path = {'modified_datetime': props_2['modified_datetime'], 
                          'filesize_bytes': props_2['filesize_bytes']}
        
        is_same = io.is_same_file(file_meta_data_1=props_1_no_path, file_meta_data_2=props_2_no_path)
        assert is_same is False
        
        # Create a second file with different size
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_file_2 = f.name
            f.write("Hello World!!!")  # Different size: 14 bytes
        
        try:
            # Get properties of second file
            props_3 = io.get_file_properties(temp_file_2)
            
            assert props_3['filesize_bytes'] == 14
            
            # Files should NOT be the same due to different sizes
            is_same = io.is_same_file(file_meta_data_1=props_2, file_meta_data_2=props_3)
            assert is_same is False
            
            # Also test by passing paths directly
            is_same = io.is_same_file(path_1=temp_file_1, path_2=temp_file_2)
            assert is_same is False
            
            # Test same file comparison (should be True)
            is_same = io.is_same_file(path_1=temp_file_1, path_2=temp_file_1)
            assert is_same is True
            
        finally:
            # Clean up second temp file
            if os.path.exists(temp_file_2):
                os.remove(temp_file_2)
    
    finally:
        # Clean up first temp file
        if os.path.exists(temp_file_1):
            os.remove(temp_file_1)


def test_file_properties_with_parquet():
    """Test file properties with actual parquet files (more realistic scenario)"""
    import tempfile
    import os
    import time
    
    io = IOEngine()
    
    # Create a test DataFrame
    df1 = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    
    df2 = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8],  # More rows = larger file
        'B': [10, 20, 30, 40, 50, 60, 70, 80]
    })
    
    # Create temporary parquet files
    temp_dir = tempfile.gettempdir()
    temp_file_1 = os.path.join(temp_dir, 'test_parquet_1.parquet')
    temp_file_2 = os.path.join(temp_dir, 'test_parquet_2.parquet')
    
    try:
        # Write first parquet file
        io.to_parquet(df1, temp_file_1)
        
        # Get initial properties
        props_1_initial = io.get_file_properties(temp_file_1)
        assert props_1_initial is not None
        initial_size = props_1_initial['filesize_bytes']
        assert initial_size > 0
        
        # Sleep to ensure time difference
        time.sleep(0.1)
        
        # Touch the file
        success = io.touch_file(temp_file_1)
        assert success is True
        
        # Get properties after touch
        props_1_touched = io.get_file_properties(temp_file_1)
        
        # Verify modification time changed but size didn't
        assert props_1_touched['modified_datetime'] > props_1_initial['modified_datetime']
        assert props_1_touched['filesize_bytes'] == initial_size
        
        # Write second parquet file with more data (different size)
        io.to_parquet(df2, temp_file_2)
        props_2 = io.get_file_properties(temp_file_2)
        
        # Second file should be larger
        assert props_2['filesize_bytes'] > props_1_touched['filesize_bytes']
        
        # Files should not be the same
        is_same = io.is_same_file(path_1=temp_file_1, path_2=temp_file_2)
        assert is_same is False
        
    finally:
        # Clean up
        for f in [temp_file_1, temp_file_2]:
            if os.path.exists(f):
                os.remove(f)


def test_timezone_independence():
    """Test that file properties work correctly regardless of system timezone"""
    import tempfile
    import os
    import datetime as dt
    
    io = IOEngine()
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_file = f.name
        f.write("Timezone test")
    
    try:
        # Get file properties
        props = io.get_file_properties(temp_file)
        
        # Verify the datetime is timezone-aware and in UTC
        assert props is not None
        modified_dt = props['modified_datetime']
        
        # Check it's timezone-aware (has tzinfo)
        assert modified_dt.tzinfo is not None, "Datetime should be timezone-aware"
        
        # Check it's in UTC
        assert modified_dt.tzinfo == dt.timezone.utc, "Datetime should be in UTC timezone"
        
        # Verify it's close to current time (within 5 seconds)
        current_utc = dt.datetime.now(tz=dt.timezone.utc)
        time_diff = abs((current_utc - modified_dt).total_seconds())
        assert time_diff < 5, f"File modification time should be recent, but differs by {time_diff} seconds"
        
        # Get the raw stat info for comparison
        stat_info = os.stat(temp_file)
        
        # Manually convert using the same method to verify consistency
        manual_dt = dt.datetime.fromtimestamp(stat_info.st_mtime, tz=dt.timezone.utc)
        
        # Should match exactly
        assert modified_dt == manual_dt, "Conversion should be consistent"
        
        # Verify the timestamp is very close (allowing for floating-point precision)
        # This ensures the method works the same on any system timezone
        timestamp_diff = abs(props['modified_datetime'].timestamp() - stat_info.st_mtime)
        assert timestamp_diff < 0.001, \
            f"Timestamp should match the original st_mtime (diff: {timestamp_diff})"
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == '__main__':
    pytest.main()
