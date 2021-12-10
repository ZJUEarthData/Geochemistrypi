def basic_statistic(data):
    print("Some basic statistic information of the designated data set:")
    print(data.describe())


def is_null_value(data):
    print("Check which column has null values:")
    print(data.isnull().any())

