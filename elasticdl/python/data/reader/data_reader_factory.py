# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from elasticdl.python.common.constants import MaxComputeConfig, ReaderType
from elasticdl.python.data.odps_io import is_odps_configured
from elasticdl.python.data.reader.csv_reader import CSVDataReader
from elasticdl.python.data.reader.odps_reader import ODPSDataReader
from elasticdl.python.data.reader.recordio_reader import RecordIODataReader


def create_data_reader(data_origin, records_per_task=None, **kwargs):
    """Create a data reader to read records
    Args:
        data_origin: The origin of the data, e.g. location to files,
            table name in the database, etc.
        records_per_task: The number of records to create a task
        kwargs: data reader params, the supported keys are
            "columns", "partition", "reader_type"
    """
    reader_type = kwargs.get("reader_type", None)
    if reader_type is None:
        if is_odps_configured():
            return ODPSDataReader(
                project=os.environ[MaxComputeConfig.PROJECT_NAME],
                access_id=os.environ[MaxComputeConfig.ACCESS_ID],
                access_key=os.environ[MaxComputeConfig.ACCESS_KEY],
                table=data_origin,
                endpoint=os.environ.get(MaxComputeConfig.ENDPOINT),
                tunnel_endpoint=os.environ.get(
                    MaxComputeConfig.TUNNEL_ENDPOINT, None
                ),
                records_per_task=records_per_task,
                **kwargs,
            )
        elif data_origin and data_origin.endswith(".csv"):
            return CSVDataReader(data_dir=data_origin, **kwargs)
        else:
            return RecordIODataReader(data_dir=data_origin)
    elif reader_type == ReaderType.CSV_READER:
        return CSVDataReader(data_dir=data_origin, **kwargs)
    elif reader_type == ReaderType.ODPS_READER:
        if not is_odps_configured:
            raise ValueError(
                "MAXCOMPUTE_AK, MAXCOMPUTE_SK and MAXCOMPUTE_PROJECT ",
                "must be configured in envs",
            )
        return ODPSDataReader(
            project=os.environ[MaxComputeConfig.PROJECT_NAME],
            access_id=os.environ[MaxComputeConfig.ACCESS_ID],
            access_key=os.environ[MaxComputeConfig.ACCESS_KEY],
            table=data_origin,
            endpoint=os.environ.get(MaxComputeConfig.ENDPOINT),
            records_per_task=records_per_task,
            **kwargs,
        )
    elif reader_type == ReaderType.RECORDIO_READER:
        return RecordIODataReader(data_dir=data_origin)
    else:
        raise ValueError(
            "The reader type {} is not supported".format(reader_type)
        )
