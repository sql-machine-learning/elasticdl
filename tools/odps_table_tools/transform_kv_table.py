import argparse
import time

import odps
from odps import options, ODPS

options.log_view_host = "http://logview.odps.aliyun-inc.com:8080"

UDF_CLASS_NAME = "KVFlatter"
ANALYZE_FEATURE_RECORDS_COUNT = 100

INTER_KV_SEPARATOR = ":"
INTRA_KV_SEPARATOR = ","

TRANSFORM_SQL_TEMPLATE = "CREATE TABLE IF NOT EXISTS {output_table} LIFECYCLE 7 AS \n\
    SELECT \n\
        {udf} \n\
    FROM {input_table}"


def get_feature_names(
    odps_entry, table_name, partition, kv_column, inter_kv_sep, intra_kv_sep
):
    """ Parse the feature names from records the a table
    Args:
        table_name: ODPS table name
        partition (string): The table partition
        kv_column (string): The key-value column name

    Returns:
        list: A list with feature names
    """
    source_kv_table = odps_entry.get_table(table_name)
    key_names = set()
    for record in source_kv_table.head(
        ANALYZE_FEATURE_RECORDS_COUNT, partition=partition
    ):
        kv_dict = parse_kv_string_to_dict(
            record[kv_column], inter_kv_sep, intra_kv_sep
        )
        key_names.update(kv_dict.keys())
    return sorted(key_names)


def parse_kv_string_to_dict(kvs_string, inter_kv_sep, intra_kv_sep):
    """Parse a kv string to a dict. For example,
    "key1:value1,key2:value2" => {key1: value1, key2: value2}
    """
    kv_dict = {}
    kv_pairs = kvs_string.split(inter_kv_sep)
    for kv in kv_pairs:
        key_and_value = kv.split(intra_kv_sep)
        if len(key_and_value) == 2:
            kv_dict[key_and_value[0]] = key_and_value[1]

    return kv_dict


def generate_sql(
    input_table,
    input_table_partition,
    output_table,
    output_columns,
    kv_column,
    udf_function,
    append_columns,
    inter_kv_sep,
    intra_kv_sep,
):
    """Generate an ODPS SQL to transform the table
    Args:
        input_table: input table name
        input_table_partition: input table partition
        output_table: output table name
        output_columns (list): feature names
        kv_column: kv column name
        udf_function: udf function name
        append_columns (list): Append column names.
        inter_kv_sep: Inter separator in the Key-value pair string.
        intra_kv_sep: Intra separator int Ker-value pairs string.
    """
    feature_names_str = ",".join(output_columns)
    output_columns.extend(append_columns)
    output_columns_str = ",".join(output_columns)
    input_columns = [kv_column]
    input_columns.extend(append_columns)
    input_columns_str = ",".join(input_columns)

    udf = """{udf}({input_col_str},
    "{features_str}", "{inter_sep}", "{intra_sep}")
    as ({output_col_str})""".format(
        udf=udf_function,
        input_col_str=input_columns_str,
        features_str=feature_names_str,
        output_col_str=output_columns_str,
        inter_sep=inter_kv_sep,
        intra_sep=intra_kv_sep,
    )

    sql = TRANSFORM_SQL_TEMPLATE.format(
        output_table=output_table, udf=udf, input_table=input_table,
    )
    if input_table_partition is not None:
        sql += " where {}".format(input_table_partition)
    return sql


def exec_sql(odps_entry, sql):
    print("====> execute_sql: " + sql)
    instance = odps_entry.run_sql(sql)
    print("====> logview: " + instance.get_logview_address())
    instance.wait_for_success()


def create_udf_function(odps_entry, udf_file_path):
    udf_resource = "sqlflow_flat_{}.py".format(int(time.time()))
    udf_function = "sqlflow_flat_func_{}".format(int(time.time()))

    delete_udf_resource(odps_entry, udf_resource)
    resource = odps_entry.create_resource(
        udf_resource, type="py", file_obj=open(udf_file_path)
    )
    print("Create python resource: {}".format(udf_resource))

    delete_udf_function(odps_entry, udf_function)
    class_type = udf_resource[0:-2] + UDF_CLASS_NAME
    odps_entry.create_function(
        udf_function, class_type=class_type, resources=[resource]
    )

    return udf_resource, udf_function


def delete_udf_resource(odps_entry, udf_resource):
    try:
        py_resource = odps_entry.get_resource(udf_resource)
        if py_resource:
            py_resource.drop()
    except odps.errors.NoSuchObject:
        pass
    finally:
        print("Drop resource if exists {}".format(udf_resource))


def delete_udf_function(odps_entry, udf_function):
    try:
        function = odps_entry.get_function(udf_function)
        function.drop()
    except odps.errors.NoSuchObject:
        pass
    finally:
        print("Drop function is exists {}".format(udf_function))


def flat_to_wide_table(
    odps_entry,
    input_table,
    kv_column,
    output_table,
    udf_file_path,
    inter_kv_sep,
    intra_kv_sep,
    input_table_partition=None,
    append_columns=None,
):
    """Transform the kv column to wide table
    Args:
        odps_entry: ODPS entry instance
        input_table: The input table name.
        kv_column: The key-value pairs column name.
        output_table: The output table name.
        udf_file_path: The python udf file path.
        input_table_partition: The input table partition.
        append_columns: The columns appended to output table.
        inter_kv_sep: Inter separator in the Key-value pair string.
        intra_kv_sep: Intra separator int Ker-value pairs string.
    """
    try:
        udf_resource, udf_function = create_udf_function(
            odps_entry, udf_file_path
        )
        odps_entry.delete_table(output_table, if_exists=True)
        feature_names = get_feature_names(
            odps_entry,
            input_table,
            input_table_partition,
            kv_column,
            inter_kv_sep,
            intra_kv_sep,
        )
        sql = generate_sql(
            input_table,
            input_table_partition,
            output_table,
            feature_names,
            kv_column,
            udf_function,
            append_columns,
            inter_kv_sep,
            intra_kv_sep,
        )
        exec_sql(odps_entry, sql)
    finally:
        delete_udf_function(odps_entry, udf_function)
        delete_udf_resource(odps_entry, udf_resource)


def add_params(parser):
    parser.add_argument(
        "--udf_file_path",
        default="",
        type=str,
        help="The path of udf python file",
        required=True,
    )
    parser.add_argument(
        "--input_table",
        default="",
        type=str,
        help="The input odps table name",
        required=True,
    )
    parser.add_argument(
        "--input_table_partition",
        default=None,
        type=str,
        help="The partition of input table",
    )
    parser.add_argument(
        "--kv_column",
        default="",
        type=str,
        help="The name of kv column to transform",
        required=True,
    )
    parser.add_argument(
        "--output_table",
        default="",
        type=str,
        help="The output table name",
        required=True,
    )
    parser.add_argument(
        "--append_columns",
        default=None,
        type=str,
        help="the append columns to output table like 'id,label'",
    )
    parser.add_argument(
        "--inter_kv_separator",
        default=INTER_KV_SEPARATOR,
        type=str,
        help="The inter key-value separator in a key-value pair string",
    )
    parser.add_argument(
        "--intra_kv_separator",
        default=INTRA_KV_SEPARATOR,
        type=str,
        help="The intra key-value pairs separator'",
    )
    parser.add_argument(
        "--MAXCOMPUTE_AK",
        default=None,
        required=True,
        type=str,
        help="The intra key-value pairs separator'",
    )
    parser.add_argument(
        "--MAXCOMPUTE_SK",
        default=None,
        required=True,
        type=str,
        help="The intra key-value pairs separator'",
    )
    parser.add_argument(
        "--MAXCOMPUTE_PROJECT",
        default=None,
        required=True,
        type=str,
        help="The intra key-value pairs separator'",
    )
    parser.add_argument(
        "--MAXCOMPUTE_ENDPOINT",
        default=None,
        type=str,
        help="The intra key-value pairs separator'",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_params(parser)
    args, _ = parser.parse_known_args()

    odps_entry = ODPS(
        access_id=args.MAXCOMPUTE_AK,
        secret_access_key=args.MAXCOMPUTE_SK,
        project=args.MAXCOMPUTE_PROJECT,
        endpoint=args.MAXCOMPUTE_ENDPOINT
    )

    append_columns = (
        args.append_columns.strip().split(",")
        if args.append_columns is not None
        else None
    )

    flat_to_wide_table(
        odps_entry,
        args.input_table,
        args.kv_column,
        args.output_table,
        args.udf_file_path,
        args.inter_kv_separator,
        args.intra_kv_separator,
        args.input_table_partition,
        append_columns,
    )
