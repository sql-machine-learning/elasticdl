import os

from elasticdl.python.common.constants import ODPSConfig
from elasticdl.python.data.odps_io import is_odps_configured
from elasticdl.python.data.reader.csv_reader import CSVDataReader
from elasticdl.python.data.reader.odps_reader import ODPSDataReader
from elasticdl.python.data.reader.recordio_reader import RecordIODataReader


def create_data_reader(data_origin, records_per_task=None, **kwargs):
    if is_odps_configured():
        return ODPSDataReader(
            project=os.environ[ODPSConfig.PROJECT_NAME],
            access_id=os.environ[ODPSConfig.ACCESS_ID],
            access_key=os.environ[ODPSConfig.ACCESS_KEY],
            table=data_origin,
            endpoint=os.environ.get(ODPSConfig.ENDPOINT),
            records_per_task=records_per_task,
            **kwargs,
        )
    elif data_origin and data_origin.endswith(".csv"):
        return CSVDataReader(data_dir=data_origin)
    else:
        return RecordIODataReader(data_dir=data_origin)
