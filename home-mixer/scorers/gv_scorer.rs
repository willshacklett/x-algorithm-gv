use crate::candidate_pipeline::candidate::{PhoenixScores, PostCandidate};
use crate::candidate_pipeline::query::ScoredPostsQuery;
use crate::params as p;
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;

pub struct GvScorer;

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for GvScorer {
    #[xai_stats_macro::receive_stats]
    async fn score(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<PostCandidate>, String> {
        let scored = candidates
            .iter()
            .map(|c| {
                // WeightedScorer has already set weighted_score; we simply adjust it.
                let base = c.weighted_score.unwrap_or(0.0);

                let gv = Self::compute_gv(&c.phoenix_scores);
                let mult = Self::gv_multiplier(gv);

                PostCandidate {
                    weighted_score: Some(base * mult),
                    ..Default::default()
                }
            })
            .collect();

        Ok(scored)
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.weighted_score = scored.weighted_score;
    }
}

impl GvScorer {
    fn get(x: Option<f64>) -> f64 {
        x.unwrap_or(0.0)
    }

    /// Gv in [0,1] derived from Phoenix predictions.
    ///
    /// Idea:
    /// - "Value" signals add
    /// - "Regret/risk" signals subtract
    /// Then sigmoid -> 0..1
    fn compute_gv(s: &PhoenixScores) -> f64 {
        let pos =
            Self::get(s.favorite_score) * p::GV_FAVORITE_W +
            Self::get(s.reply_score) * p::GV_REPLY_W +
            Self::get(s.retweet_score) * p::GV_RETWEET_W +
            Self::get(s.share_score) * p::GV_SHARE_W +
            Self::get(s.dwell_score) * p::GV_DWELL_W +
            Self::get(s.quote_score) * p::GV_QUOTE_W +
            Self::get(s.follow_author_score) * p::GV_FOLLOW_AUTHOR_W;

        let neg =
            Self::get(s.not_interested_score) * p::GV_NOT_INTERESTED_W +
            Self::get(s.mute_author_score) * p::GV_MUTE_AUTHOR_W +
            Self::get(s.block_author_score) * p::GV_BLOCK_AUTHOR_W +
            Self::get(s.report_score) * p::GV_REPORT_W;

        let raw = (pos - neg) + p::GV_BIAS;

        // Sigmoid squashing
        1.0 / (1.0 + (-(p::GV_SIGMOID_K * raw)).exp())
    }

    /// Convert Gv into a multiplicative factor that adjusts the existing weighted_score.
    ///
    /// - gv_floor keeps exploration alive (never fully zeroes out)
    /// - gv_strength controls how strongly Gv influences rank
    fn gv_multiplier(gv: f64) -> f64 {
        let gv = gv.clamp(0.0, 1.0);
        let shaped = p::GV_FLOOR + (1.0 - p::GV_FLOOR) * gv; // still 0..1
        1.0 + p::GV_STRENGTH * (shaped - 0.5) * 2.0 // maps [0,1] -> [-1,1] then scales
    }
}
