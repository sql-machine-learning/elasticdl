import os

from elasticdl.python.common.constants import ODPSConfig, ReaderType
from elasticdl.python.data.odps_io import is_odps_configured
from elasticdl.python.data.reader.csv_reader import CSVDataReader
from elasticdl.python.data.reader.odps_reader import ODPSDataReader
from elasticdl.python.data.reader.recordio_reader import RecordIODataReader


def create_data_reader(data_origin, records_per_task=None, **kwargs):
    """Create a data reader to read records
    Args:
        records_per_task: The number of records to create a task
        kwargs: data reader params
    """
    reader_type = kwargs.get("reader_type", None)
    if reader_type is None:
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
            return CSVDataReader(data_dir=data_origin, **kwargs)
        else:
            return RecordIODataReader(data_dir=data_origin)
    elif reader_type == ReaderType.CSV_READER:
        return CSVDataReader(data_dir=data_origin, **kwargs)
    elif reader_type == ReaderType.ODPS_READER:
        if not is_odps_configured:
            raise ValueError(
                "ODPS_ACCESS_ID, ODPS_ACCESS_KEY and ODPS_PROJECT_NAME ",
                "must be configured in envs",
            )
        return ODPSDataReader(
            project=os.environ[ODPSConfig.PROJECT_NAME],
            access_id=os.environ[ODPSConfig.ACCESS_ID],
            access_key=os.environ[ODPSConfig.ACCESS_KEY],
            table=data_origin,
            endpoint=os.environ.get(ODPSConfig.ENDPOINT),
            records_per_task=records_per_task,
            **kwargs,
        )
    elif reader_type == ReaderType.RECORDIO_READER:
        return RecordIODataReader(data_dir=data_origin)
    else:
        raise ValueError(
            "The reader type {} is not supported".format(reader_type)
        )
