import pandas as pd

class PhasesOfPlayAggregator:
    """
    Aggregates phase-of-play data for in-possession and out-of-possession teams.

    This class:
    - Adds missing out-of-possession team info.
    - Calculates the next phase for each team.
    - Produces aggregated stats for each phase type, including counts, durations,
      spatial averages, and transition counts.
    """

    def __init__(self, phases_of_play_df):
        """
        Initialize with a DataFrame of phases of play.

        Parameters
        ----------
        phases_of_play_df : pd.DataFrame
            DataFrame containing phase-of-play information.
            Must include possession team IDs, phase types, timing, and metrics.
        """
        self.phases_of_play_df = phases_of_play_df.copy()
        required = ["match_id", "period", "frame_start", "frame_end",
                    "team_in_possession_id", "team_in_possession_shortname"]
        if self.phases_of_play_df.empty or self.phases_of_play_df[required].isna().any().any():
            raise ValueError("Phases require nonempty, non-null match/period/frame/team identities")
        if self.phases_of_play_df.duplicated(["match_id", "period", "frame_start"]).any():
            raise ValueError("Phase start frames must be unique within a match and period")
        if (self.phases_of_play_df["frame_end"] <= self.phases_of_play_df["frame_start"]).any():
            raise ValueError("Phase end frames must follow start frames")
        self._add_team_out_of_possession_info()
        self._add_next_phase()

    def _add_team_out_of_possession_info(self):
        """
        Fill in `team_out_of_possession_id` and `team_out_of_possession_shortname`
        by swapping the two in-possession teams.
        """
        opponents = {}
        for match_id, group in self.phases_of_play_df.groupby("match_id"):
            teams = group[["team_in_possession_id", "team_in_possession_shortname"]].drop_duplicates()
            if len(teams) != 2 or teams["team_in_possession_id"].nunique() != 2:
                raise ValueError(f"Match {match_id} must identify exactly two consistent teams")
            first, second = list(teams.itertuples(index=False, name=None))
            opponents[(match_id, first[0])] = second
            opponents[(match_id, second[0])] = first
        identities = [
            opponents[(row.match_id, row.team_in_possession_id)]
            for row in self.phases_of_play_df.itertuples()
        ]
        self.phases_of_play_df["team_out_of_possession_id"] = [team[0] for team in identities]
        self.phases_of_play_df["team_out_of_possession_shortname"] = [team[1] for team in identities]

    def _add_next_phase(self):
        """
        Determine the next phase for both teams based on time continuity.
        If a phase ends when another starts, record the next phase type.
        Otherwise, set 'no_next_phase'.
        """
        self.phases_of_play_df = self.phases_of_play_df.sort_values(
            by=["match_id", "period", "frame_start"]
        ).reset_index(drop=True)
        self.phases_of_play_df["team_in_possession_next_phase"] = None
        self.phases_of_play_df["team_out_of_possession_next_phase"] = None

        frame_start_to_index = {
            (row.match_id, row.period, row.frame_start): i
            for i, row in enumerate(self.phases_of_play_df.itertuples())
        }

        for i in range(len(self.phases_of_play_df) - 1):
            current_frame_end = (
                self.phases_of_play_df.loc[i, "match_id"],
                self.phases_of_play_df.loc[i, "period"],
                self.phases_of_play_df.loc[i, "frame_end"],
            )
            if current_frame_end in frame_start_to_index:
                next_idx = frame_start_to_index[current_frame_end]

                if self.phases_of_play_df.loc[i, "team_in_possession_id"] == self.phases_of_play_df.loc[next_idx, "team_in_possession_id"]:
                    self.phases_of_play_df.loc[i, "team_in_possession_next_phase"] = self.phases_of_play_df.loc[next_idx, "team_in_possession_phase_type"]
                else:
                    self.phases_of_play_df.loc[i, "team_in_possession_next_phase"] = self.phases_of_play_df.loc[next_idx, "team_out_of_possession_phase_type"]

                if self.phases_of_play_df.loc[i, "team_out_of_possession_id"] == self.phases_of_play_df.loc[next_idx, "team_out_of_possession_id"]:
                    self.phases_of_play_df.loc[i, "team_out_of_possession_next_phase"] = self.phases_of_play_df.loc[next_idx, "team_out_of_possession_phase_type"]
                else:
                    self.phases_of_play_df.loc[i, "team_out_of_possession_next_phase"] = self.phases_of_play_df.loc[next_idx, "team_in_possession_phase_type"]

        self.phases_of_play_df.loc[self.phases_of_play_df['team_in_possession_next_phase'].isnull(), 'team_in_possession_next_phase'] = 'no_next_phase'
        self.phases_of_play_df.loc[self.phases_of_play_df['team_out_of_possession_next_phase'].isnull(), 'team_out_of_possession_next_phase'] = 'no_next_phase'

    def get_out_of_possession_aggregates(self):
        """
        Aggregate stats for out-of-possession phases.

        Returns
        -------
        pd.DataFrame
            One row per (match, team, phase type) with counts, total time,
            event counts, average positions, and next-phase counts.
        """
        group_by = ['match_id', 'team_out_of_possession_id', 'team_out_of_possession_shortname', 'team_out_of_possession_phase_type']

        out_of_possession_phase_aggs = self.phases_of_play_df.groupby(group_by).agg(
            count=('team_out_of_possession_phase_type', 'count'),
            total_time=('duration', 'sum'),
            count_player_possessions=('n_player_possessions_in_phase', 'sum'),
            count_possession_lost_in_phase=('team_possession_loss_in_phase', 'sum'),
            count_possession_lead_to_shot=('team_possession_lead_to_shot', 'sum'),
            count_possession_lead_to_goal=('team_possession_lead_to_goal', 'sum'),
            avg_start_width=('team_out_of_possession_width_start', 'mean'),
            avg_start_length=('team_out_of_possession_length_start', 'mean'),
            avg_end_width=('team_out_of_possession_width_end', 'mean'),
            avg_end_length=('team_out_of_possession_length_end', 'mean')
        ).reset_index()

        out_of_possession_phase_aggs = self._attach_transition_counts(
            out_of_possession_phase_aggs, "team_out_of_possession"
        )

        out_of_possession_phase_aggs = out_of_possession_phase_aggs.set_index([
            'match_id', 'team_out_of_possession_id', 'team_out_of_possession_shortname', 'team_out_of_possession_phase_type'
        ]).unstack(['team_out_of_possession_phase_type'])
        out_of_possession_phase_aggs.columns = ['{}_{}'.format(m, c) for m, c in out_of_possession_phase_aggs.columns]
        out_of_possession_phase_aggs = out_of_possession_phase_aggs.reset_index()

        metric_columns = []
        for phase in ['low_block', 'medium_block', 'high_block', 'defending_transition', 'defending_quick_break',
                      'defending_direct', 'chaotic', 'defending_set_play']:

            for metric in ['count', 'total_time', 'count_player_possessions', 'count_possession_lost_in_phase',
                           'count_possession_lead_to_shot', 'count_possession_lead_to_goal', 'avg_start_width',
                           'avg_start_length', 'avg_end_width', 'avg_end_length', 'count_into_build_up_from',
                           'count_into_create_from', 'count_into_finish_from', 'count_into_transition_from',
                           'count_into_quick_break_from', 'count_into_direct_from', 'count_into_chaotic_from',
                           'count_into_low_block_from', 'count_into_medium_block_from', 'count_into_high_block_from',
                           'count_into_defending_quick_break_from', 'count_into_defending_transition_from',
                           'count_into_defending_direct_from']:
                metric_columns.append(f'{metric}_{phase}')

        return out_of_possession_phase_aggs.reindex(columns=group_by[:3] + metric_columns)

    def get_in_possession_aggregates(self):
        """
        Aggregate stats for in-possession phases.

        Returns
        -------
        pd.DataFrame
            One row per (match, team, phase type) with counts, total time,
            event counts, average positions, and next-phase counts.
        """
        group_by = ['match_id', 'team_in_possession_id', 'team_in_possession_shortname', 'team_in_possession_phase_type']

        in_possession_phase_aggs = self.phases_of_play_df.groupby(group_by).agg(
            count=('team_in_possession_phase_type', 'count'),
            total_time=('duration', 'sum'),
            count_player_possessions=('n_player_possessions_in_phase', 'sum'),
            count_possession_lost_in_phase=('team_possession_loss_in_phase', 'sum'),
            count_possession_lead_to_shot=('team_possession_lead_to_shot', 'sum'),
            count_possession_lead_to_goal=('team_possession_lead_to_goal', 'sum'),
            avg_start_width=('team_in_possession_width_start', 'mean'),
            avg_start_length=('team_in_possession_length_start', 'mean'),
            avg_end_width=('team_in_possession_width_end', 'mean'),
            avg_end_length=('team_in_possession_length_end', 'mean')
        ).reset_index()

        in_possession_phase_aggs = self._attach_transition_counts(
            in_possession_phase_aggs, "team_in_possession"
        )

        in_possession_phase_aggs = in_possession_phase_aggs.set_index([
            'match_id', 'team_in_possession_id', 'team_in_possession_shortname', 'team_in_possession_phase_type'
        ]).unstack(['team_in_possession_phase_type'])
        in_possession_phase_aggs.columns = ['{}_{}'.format(m, c) for m, c in in_possession_phase_aggs.columns]
        in_possession_phase_aggs = in_possession_phase_aggs.reset_index()

        metric_columns = []
        for phase in ['build_up', 'create', 'finish', 'transition', 'quick_break', 'direct', 'chaotic', 'set_play']:
            for metric in ['count', 'total_time', 'count_player_possessions', 'count_possession_lost_in_phase',
                           'count_possession_lead_to_shot', 'count_possession_lead_to_goal', 'avg_start_width',
                           'avg_start_length', 'avg_end_width', 'avg_end_length', 'count_into_build_up_from',
                           'count_into_create_from', 'count_into_finish_from', 'count_into_transition_from',
                           'count_into_quick_break_from', 'count_into_direct_from', 'count_into_chaotic_from',
                           'count_into_low_block_from', 'count_into_medium_block_from', 'count_into_high_block_from',
                           'count_into_defending_quick_break_from', 'count_into_defending_transition_from',
                           'count_into_defending_direct_from']:

                metric_columns.append(f'{metric}_{phase}')

        return in_possession_phase_aggs.reindex(columns=group_by[:3] + metric_columns)

    def _attach_transition_counts(self, aggregates, team_prefix):
        keys = ["match_id", f"{team_prefix}_id", f"{team_prefix}_phase_type"]
        next_column = f"{team_prefix}_next_phase"
        counts = self.phases_of_play_df.groupby(keys + [next_column]).size().unstack(
            next_column, fill_value=0
        )
        counts.columns = [f"count_into_{phase}_from" for phase in counts.columns]
        return aggregates.merge(counts.reset_index(), on=keys, how="left", validate="many_to_one")
