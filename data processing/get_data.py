
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer



def get_data(base_plan="../data processing/berth_assignments.csv",
            human_adjusted="../data processing/augmented_berth_assignments.csv",
            data_input_for_model="../data processing/estimated_final.csv"
            ):
    """
    returns 3 values:
    1) dataframe input for AI
    2) feature_cols (input column names)
    3) label_cols (output target column names)
    """

    # df = pandas.Dataframe
    base_plan_df = pd.read_csv(base_plan)

    human_adjusted_df = pd.read_csv(human_adjusted)

    data_input_for_model = pd.read_csv(data_input_for_model)


    base_plan_df = base_plan_df.rename(columns={
                                                "BERTH" : "BASE_BERTH",
                                                 "ASSIGNED_START" : "BASE_ETA",
                                                  "ESTIMATED_END" : "BASE_ETD"
                                                  })
    
    human_adjusted_df = human_adjusted_df.rename(columns={
                                                            "BERTH" : "H_BERTH",
                                                            "ADJUSTED_START" : "H_ETA",
                                                            "ADJUSTED_END" : "H_ETD"
                                                            })



    # merge data into 1 thing
    merged_df = base_plan_df.merge(human_adjusted_df, on='CALL_SIGN')

    merged_df = merged_df.merge(data_input_for_model, on="CALL_SIGN")

    # TODO
    # ETD_TIME, ETA_TIME ?
    feature_cols = ["BASE_BERTH", "BASE_ETA", "BASE_ETD", "AGENT_NAME", "VESSEL_NAME", "VESSEL_MAX_DRAFT", "WEATHER_IMPACT_PCT", "CONGESTION_IMPACT_PCT", "EFFECTIVENESS_SCORE", "RELIABILITY_SCORE", "WORK_ENV_SCORE"]
    label_cols = ["H_BERTH", "H_ETA", "H_ETD"]



    # TODO 
    # convert to num columns:
    #   - BASE_BERTH
    #   - AGENT_NAME
    #   - VESSEL_NAME
    #   - FLAG
    #   - H_BERTH

    le = LabelEncoder
    string_col_names = ["BASE_BERTH", "AGENT_NAME", "VESSEL_NAME", "H_BERTH"]
    for col_name in string_col_names:
        # print(col_name)
        # print(merged_df[col_name])
        merged_df[col_name] = le.fit_transform(le, merged_df[col_name])
        # print(merged_df[col_name])


    # date to num 
    date_cols = ["BASE_ETA", "BASE_ETD", "H_ETA", "H_ETD"]
    for date_col in date_cols:
        merged_df[date_col] = pd.to_datetime(merged_df[date_col])
        merged_df[date_col] = merged_df[date_col].apply(lambda x: x.timestamp() - pd.to_datetime("2025-05-09 0:0:0").timestamp())


    
    # imp = SimpleImputer(strategy="median")

    # merged_df[feature_cols] = imp.fit_transform(merged_df[feature_cols])

    # merged_df[feature_cols]= merged_df[feature_cols].dropna(how="all")
    # merged_df[label_cols]= merged_df[label_cols].dropna(how="all")

    return merged_df, feature_cols, label_cols