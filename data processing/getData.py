
import pandas as pd

def getData(base_plan="berth_assignments.csv",
            human_adjusted="augmented_berth_assignments.csv",
            data_input_for_model="estimated_final.csv"
            ):
    """
    3 returns:
    1) base plan
    2) human-adjusted plan (correct answer that ML is trying to get close to)
    3) input for model (including base plan)
    """

    # df = pandas.Dataframe
    base_plan_df = pd.read_csv(base_plan)

    human_adjusted_df = pd.read_csv(human_adjusted)

    data_input_for_model = pd.read_csv(data_input_for_model)

    # TODO