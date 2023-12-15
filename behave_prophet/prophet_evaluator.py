import pandas as pd
import numpy as np
from tqdm import tqdm
from prophet import Prophet
from dataclasses import dataclass
from sqlalchemy import create_engine, text
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from behave_prophet.colors import colors

import logging
import cmdstanpy


def behave_logger():
    """Set up the package logger"""

    logger = logging.getLogger("behave")
    logger.setLevel(logging.INFO)

    # Create a handler to suppress cmdstanpy INFO messages
    cmdstanpy_logger = logging.getLogger("cmdstanpy")
    cmdstanpy_logger.setLevel(logging.WARNING)
    cmdstanpy_logger.addHandler(logging.NullHandler())

    return logger


# Seting up the package logger
behave_logging = behave_logger()


con = create_engine("postgresql://postgres:postgres@localhost:5432/behave", echo=False)


@dataclass
class ProphetEvaluator:
    """
    A class to evaluate the behavior of a forecasting model in a simulated environment.

    ...

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the timeseries.
    target : str
        The name of the target column.
    date : str
        The name of the date column.
    testtype : str
        The type of the test. Currently only "structural" is supported.
    sim_runs : int
        The number of simulations to run. Default is 10.
    sim_value : float
        The value to multiply the target with in the simulation. Default is 1.2.
    outlier_injection : float
        The value to multiply the target with in the simulation. Default is 1.8.

    Attributes
    ----------
    Methods
    -------
    evaluate()
        Perform the simulation and forecast evaluation.
    plot_interventions()
        Visualize the randomly selected intervention points and the simulated timeseries.
    plot_summary()
        Provide a global report on the simulation results.
    plot_detail()
        Visualize the simulated timeseries & forecasts for each evaluation step for a single run.
    plot_mapes()
        Visualize the mapes for each evaluation step for a single run.
    """

    df: pd.DataFrame
    target: str
    date: str
    testtype: str
    sim_runs: int = 3
    sim_value: float = 1.2
    outlier_injection: float = 1.8
    jobid: str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    run: int = 0

    def __prep_data(self):
        """
        Prepare the data for the simulation.
        """
        df = self.df.copy()
        df.rename(columns={self.date: "ds", self.target: "y"}, inplace=True)

        if df.y[-12:].var() == 0:
            raise Exception(
                "Check your timeseries. The last 12 months of your timeseries have 0 variance."
            )
        df.ds = pd.to_datetime(df.ds)
        ts_start = df.ds.min()
        ts_end = df.ds.max()
        sim_range_start = ts_start + pd.DateOffset(months=24)
        sim_range_end = ts_end - pd.DateOffset(months=24)
        sim_range = pd.date_range(sim_range_start, sim_range_end, freq="M")

        if (len(sim_range) == 0) or (len(sim_range) < self.sim_runs):
            raise Exception(
                "The provided timeseries seems to be too short for the behavior testing setting. Use longer timeseries - in a monthly setting the timeseries should be about 5 years long."
            )

        sim_dates = np.random.choice(sim_range, self.sim_runs, replace=False)

        dfs = []
        for i, date in enumerate(sim_dates):
            df_sim = df.copy()
            df_sim = df_sim.assign(
                y_original=df_sim.y,
                sim_date=date,
                run=i,
            )

            dfs.append(df_sim)

        self.sim_dates = pd.to_datetime(sim_dates)
        self._sim_dfs = dfs

    def __prep_training_data(self) -> dict:
        """
        Prepare the training data for the simulation steps.
        """
        runs = {}
        sim_dates = self.sim_dates
        dfs = self._sim_dfs
        for i, date in enumerate(sim_dates):
            df = dfs[i]
            eval_start = date - pd.DateOffset(months=12)
            eval_range = pd.date_range(eval_start, periods=24, freq="M")
            train_dfs = {}
            for j, training_cutoff in enumerate(eval_range):
                train_df = df.loc[df.ds < training_cutoff, ["ds", "y", "y_original"]]
                if training_cutoff >= date:
                    train_df.loc[train_df.ds >= date, "y"] = (
                        train_df.loc[train_df.ds >= date, "y"] * self.sim_value
                    )
                train_dfs[f"eval_{j}"] = train_df

            runs[i] = train_dfs

        self.train_runs = runs

    def __to_db(self, predictions, run):
        predictions.columns = predictions.columns.str.replace("_", "")
        predictions.rename(columns={"forecastinit": "fcstinit"}, inplace=True)
        predictions = predictions.assign(
            jobid=self.jobid,
            testtype=self.testtype,
            run=run,
        )
        predictions.to_sql("behave", con, if_exists="append", index=False)

    def __get_predictions(self) -> pd.DataFrame:
        train_runs = self.train_runs
        ref_dfs = self._sim_dfs

        for run, steps in tqdm(train_runs.items(), desc=" simulation runs", position=0):
            ref_df = ref_dfs[run].copy().loc[:, ["ds", "y_original", "sim_date"]]
            predictions = {}

            for name, df in tqdm(steps.items(), desc=" evaluation steps", position=1):
                end_training_date = df.iloc[-1, 0] + pd.DateOffset(months=1)
                test_range = pd.date_range(end_training_date, periods=12, freq="M")
                test = pd.DataFrame({"ds": test_range})

                m = Prophet()
                m.fit(df[["ds", "y"]])

                yhats = m.predict(test)
                yhats["fcstinit"] = test_range[0]
                yhats = yhats.merge(ref_df, on="ds", how="left")

                eval_step = str(name[5:])
                yhats["eval_step"] = eval_step

                predictions[name] = yhats

            predictions = pd.concat(predictions.values(), ignore_index=True)
            predictions = self.__post_processing(predictions)
            self.__to_db(predictions, run)

    def __post_processing(self, pred_df) -> pd.DataFrame:
        pred_df["y"] = np.where(
            pred_df.fcstinit >= pred_df.sim_date,
            pred_df.y_original * self.sim_value,
            pred_df.y_original,
        )
        pred_df["sim"] = np.where(pred_df.y == pred_df.y_original, 0, 1)

        return pred_df

    def __get_errors(self) -> pd.DataFrame:
        calculate_mapes_sql = text(
            f"UPDATE behave SET error = yhat - y WHERE jobid='{self.jobid}';"
        )
        with con.connect() as connection:
            connection.execute(calculate_mapes_sql)
            connection.commit()

    def __get_data(self) -> pd.DataFrame:
        """
        Fetch data from the database for the current jobid.
        """
        get_data_sql = text(f"SELECT * FROM behave WHERE jobid='{self.jobid}';")
        with con.connect() as connection:
            data = pd.read_sql(get_data_sql, connection)
        return data

    def evaluate(self):
        """
        Perform the simulation and forecast evaluation.

        Returns:
            None
        """

        self.__prep_data()
        self.__prep_training_data()
        self.__get_predictions()
        self.__get_errors()

    def plot_interventions(self) -> go.Figure:
        """
        Visualize the randomly selected intervention points and the simulated timeseries.

        Returns:
            fig: go.Figure
        """

        df = self.__get_data()
        df.ds = pd.to_datetime(df.ds)
        df.sort_values(by=["ds"], inplace=True)
        interventions = df.simdate.unique()

        ref = self._sim_dfs[0]

        # Create subplots
        subtitles = [f"Intervention {i}" for i in range(1, self.sim_runs + 1)]
        subtitles = ["Interventions Overview"] + subtitles
        fig = make_subplots(
            rows=self.sim_runs + 1,
            cols=1,
            subplot_titles=subtitles,
            shared_xaxes=True,
            vertical_spacing=0.008,
        )
        # Intervention overview
        overview = go.Scatter(
            x=ref.ds,
            y=ref.y_original,
            mode="lines",
            name="actuals",
            line=dict(color=colors["ACTUAL"]),
        )
        fig.add_trace(overview, row=1, col=1)

        for intervention in interventions:
            intervention = pd.to_datetime(intervention)

            fig.add_vline(
                x=intervention,
                line_width=3,
                line_dash="dash",
                line_color=colors["INTERVENTION"],
                name=f"intervention {intervention}",
                row=1,
                col=1,
            )

        # Interventions
        for i, intervention in enumerate(interventions):
            inter_df = df[
                (df.simdate == intervention)
                & (df.ds >= pd.to_datetime(intervention))
                & (df.sim == 1)
            ]
            intervention_simulated = go.Scatter(
                x=inter_df.ds,
                y=inter_df.y,
                mode="lines",
                name="simulated",
                line=dict(color=colors["SIMULATION"]),
                showlegend=False,
            )

            if i == 0:
                intervention_simulated.update(showlegend=True)

            fig.add_trace(overview.update(showlegend=False), row=i + 2, col=1)
            fig.add_trace(intervention_simulated, row=i + 2, col=1)

            fig.add_vline(
                x=pd.to_datetime(intervention),
                line_width=2,
                opacity=0.3,
                line_dash="dash",
                line_color=colors["INTERVENTION"],
                name=f"intervention {intervention}",
                row=i + 2,
                col=1,
            )

        fig.update_layout(height=1000)

        return fig

    def __get_adaption_time(self, df: pd.DataFrame) -> (np.array, pd.DataFrame):
        """
        Get the adaption time for each intervention and the mapes for each eval step.
        df: pd.DataFrame

        Returns:
            adaption_list: np.array
            mapes_df: pd.DataFrame
        """

        mapes_df = (
            df.groupby(["run", "evalstep", "sim"])
            .apply(lambda x: np.mean(np.abs(x.error) / x.y))
            .reset_index()
            .rename(columns={0: "mape"})
        )

        benchmark_df = (
            mapes_df.pipe(lambda x: x.loc[x.sim == 0])
            .groupby(["run"])
            .mape.mean()
            .reset_index()
            .rename(columns={"mape": "benchmark"})
        )

        adaption_df = (
            df.pipe(lambda x: x.loc[x.sim == 1])
            .pipe(
                lambda x: x.groupby(["run", "evalstep", "fcstinit"]).apply(
                    lambda xx: np.mean(np.abs(xx.error) / xx.y)
                )
            )
            .reset_index()
            .rename(columns={0: "mape"})
            .merge(benchmark_df, on="run")
            .pipe(lambda x: x.assign(adaption=x.mape < x.benchmark))
            .sort_values(
                by=["run", "adaption", "evalstep"], ascending=[True, False, True]
            )
        )

        # Share of evaluation steps below benchmark
        coverage = adaption_df.groupby("run").adaption.sum().values / 12

        # Check for non adaption in testing period
        nonadaption = {}
        for run in adaption_df.run.unique():
            check = adaption_df.loc[adaption_df.run == run]
            if check.adaption.sum() == 0:
                nonadaption[run] = 24

        adaption_list = adaption_df.groupby("run").head(1).evalstep.values

        if len(nonadaption) != 0:
            for run, time in nonadaption.items():
                adaption_list[run] = time

        def rename_cols(df):
            df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
            return df

        global_summary = (
            adaption_df.groupby("run")[["mape", "benchmark"]]
            .agg(["mean", "std"])
            .assign(adaption=adaption_list - 12)
            .reset_index()
            .pipe(rename_cols)
            .rename(
                columns={
                    "run_": "run",
                    "adaption_": "adaption_time",
                    "mape_mean": "sim_mape",
                    "mape_std": "sim_mape_std",
                    "benchmark_mean": "benchmark_mape",
                }
            )
            .drop(columns=["benchmark_std"])
        )

        # remove 12 pre-simulation steps for adaption time
        return adaption_list - 12, mapes_df, coverage, global_summary

    def __get_summary(self) -> (str, go.Figure):
        """
        Provide average adoption time for all runs.
        """
        df = self.__get_data()
        adaption_list, mapes_df, coverage, global_summary = self.__get_adaption_time(df)

        summary_text = (
            f"The median time to adapt to a shock is {np.median(adaption_list)} months."
        )
        summary_table = pd.DataFrame(
            data={
                "Statistic": ["Runs", "Median", "Mean", "Std", "Min", "Max"],
                "Value": [
                    len(adaption_list),
                    np.median(adaption_list),
                    np.mean(adaption_list),
                    np.std(adaption_list),
                    np.min(adaption_list),
                    np.max(adaption_list),
                ],
            }
        )

        adaption_hist = px.histogram(
            x=adaption_list,
            nbins=10,
            title="Distribution of Adaption Times",
            labels={"x": "Adaption Time"},
        )

        adaption_hist.update_xaxes(range=[0, 12])

        coverage_box = px.box(
            coverage, title="Share of Evaluation Steps Below Benchmark Per Run"
        )
        coverage_box.update_layout(xaxis_title="", yaxis_title="Share")

        mape_box = px.box(
            mapes_df,
            x="evalstep",
            y="mape",
            color="sim",
            title="Distribution of MAPEs Per Evaluation Step",
        )

        mape_box.update_layout(
            xaxis_title="Evaluation Steps",
            yaxis_title="MAPE",
            legend_title="Simulation",
        )

        return (
            summary_text,
            summary_table,
            global_summary,
            adaption_hist,
            coverage_box,
            mape_box,
        )

    def plot_summary(self) -> (go.Figure, go.Figure, go.Figure):
        """
        Provide a global report on the simulation results.

        Returns:
            adaption_hist: go.Figure
            coverage_box: go.Figure
            mape_box: go.Figure
        """

        (
            summary_text,
            summary_table,
            global_summary,
            adaption_hist,
            coverage_box,
            mape_box,
        ) = self.__get_summary()

        print(summary_text)
        print(summary_table)
        print(global_summary)

        return (
            adaption_hist,
            coverage_box,
            mape_box,
        )

    def plot_detail(self, run: int = 0) -> go.Figure:
        """
        Visualize the simulated timeseries & forecasts for each evaluation step for a single run.

        Args:
            run: int
                The run to visualize. Default is 0.

        Returns:
            fig: go.Figure

        """

        df = self.__get_data()
        run_df = df.loc[df.run == run]

        fig = make_subplots(
            rows=24,
            cols=1,
            subplot_titles=[f"Evaluation step {i}" for i in range(24)],
            shared_xaxes=True,
            vertical_spacing=0.008,
        )

        # Steps
        sim_date = run_df.simdate.unique()[0]
        ref_df = self._sim_dfs[0]
        ref_df["y"] = np.where(
            ref_df.ds >= pd.to_datetime(sim_date),
            ref_df.y_original * self.sim_value,
            ref_df.y_original,
        )

        actuals = go.Scatter(
            x=ref_df.ds,
            y=ref_df.y_original,
            mode="lines",
            name="actuals",
            line=dict(color=colors["ACTUAL"]),
            showlegend=True,
        )
        simulated = go.Scatter(
            x=ref_df.loc[ref_df.ds >= pd.to_datetime(sim_date)].ds,
            y=ref_df.loc[ref_df.ds >= pd.to_datetime(sim_date)].y,
            mode="lines",
            name="simulated",
            line=dict(color=colors["SIMULATION"]),
            showlegend=True,
        )

        for i in range(24):
            vis_df = run_df.loc[run_df.evalstep == i]
            fcstinit = vis_df.fcstinit.unique()[0]

            fcst_plot = go.Scatter(
                x=vis_df.ds,
                y=vis_df.yhat,
                mode="lines",
                name="forecast",
                line=dict(color=colors["PREDICTION"]),
                showlegend=False,
            )

            if sim_date <= fcstinit:
                if i == 13:
                    simulated.update(showlegend=False)

                fig.add_trace(simulated, row=i + 1, col=1)

            if i == 0:
                fcst_plot.update(showlegend=True)
            if i == 1:
                actuals.update(showlegend=False)

            fig.add_trace(actuals, row=i + 1, col=1)
            fig.add_trace(fcst_plot, row=i + 1, col=1)
            fig.add_vline(
                x=fcstinit,
                line_width=2,
                opacity=0.3,
                line_dash="dash",
                line_color=colors["INTERVENTION"],
                name=f"forecast origin {fcstinit}",
                row=i + 1,
                col=1,
            )

        fig.update_layout(height=10000)

        return fig

    def plot_mapes(self, run: int = 0) -> go.Figure:
        """
        Visualize the mapes for each evaluation step for a single run.

        Args:
            run: int
                The run to visualize. Default is 0.

        Returns:
            fig: go.Figure
        """

        df = self.__get_data()
        run_df = df.loc[df.run == run]

        mapes_df = (
            run_df.groupby(["evalstep", "sim"])
            .apply(lambda x: np.mean(np.abs(x.error) / x.y))
            .reset_index()
            .rename(columns={0: "mape"})
        )

        benchmark = mapes_df.loc[mapes_df.sim == 0].mape.mean()

        fig = px.line(
            mapes_df,
            x="evalstep",
            y="mape",
            color="sim",
            title="MAPEs For Each Eval Step With Benchmark",
        )

        fig.add_hline(
            y=benchmark,
            line_width=2,
            opacity=0.3,
            line_dash="dash",
        )

        fig.update_layout(
            xaxis_title="Evaluation Steps",
            yaxis_title="MAPE",
            legend_title="Simulation",
        )

        return fig
