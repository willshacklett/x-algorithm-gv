use crate::candidate_pipeline::candidate::{PhoenixScores, PostCandidate};
use crate::candidate_pipeline::query::ScoredPostsQuery;
use tonic::async_trait;
use xai_candidate_pipeline::scorer::Scorer;

//
// --------------------
// Gv Scorer (Self-Contained, Open-Source Safe)
// --------------------
// Purpose:
// Adjust the existing weighted_score using a survivability / regret-aware signal
// derived ONLY from Phoenix predictions.
// --------------------

pub struct GvScorer;

// ---------- Tunable constants (local, no params.rs) ----------

// Positive value signals
const GV_FAVORITE_W: f64 = 1.00;
const GV_REPLY_W: f64 = 0.70;
const GV_RETWEET_W: f64 = 0.80;
const GV_SHARE_W: f64 = 0.90;
const GV_DWELL_W: f64 = 0.40;
const GV_QUOTE_W: f64 = 0.50;
const GV_FOLLOW_AUTHOR_W: f64 = 1.00;

// Negative / regret signals
const GV_NOT_INTERESTED_W: f64 = 1.00;
const GV_MUTE_AUTHOR_W: f64 = 1.30;
const GV_BLOCK_AUTHOR_W: f64 = 1.60;
const GV_REPORT_W: f64 = 2.20;

// Shaping
const GV_SIGMOID_K: f64 = 3.0;
const GV_BIAS: f64 = 0.0;

// Ranking impact
const GV_FLOOR: f64 = 0.20;
const GV_STRENGTH: f64 = 0.25;

// ------------------------------------------------------------

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for GvScorer {
    async fn score(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Result<Vec<PostCandidate>, String> {
        let scored = candidates
            .iter()
            .map(|c| {
                let base = c.weighted_score.unwrap_or(0.0);
                let gv = compute_gv(&c.phoenix_scores);
                let adjusted = base * gv_multiplier(gv);

                PostCandidate {
                    weighted_score: Some(adjusted),
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

// --------------------
// Core math
// --------------------

fn get(v: Option<f64>) -> f64 {
    v.unwrap_or(0.0)
}

fn compute_gv(s: &PhoenixScores) -> f64 {
    let positive =
        get(s.favorite_score) * GV_FAVORITE_W +
        get(s.reply_score) * GV_REPLY_W +
        get(s.retweet_score) * GV_RETWEET_W +
        get(s.share_score) * GV_SHARE_W +
        get(s.dwell_score) * GV_DWELL_W +
        get(s.quote_score) * GV_QUOTE_W +
        get(s.follow_author_score) * GV_FOLLOW_AUTHOR_W;

    let negative =
        get(s.not_interested_score) * GV_NOT_INTERESTED_W +
        get(s.mute_author_score) * GV_MUTE_AUTHOR_W +
        get(s.block_author_score) * GV_BLOCK_AUTHOR_W +
        get(s.report_score) * GV_REPORT_W;

    let raw = (positive - negative) + GV_BIAS;

    1.0 / (1.0 + (-(GV_SIGMOID_K * raw)).exp())
}

fn gv_multiplier(gv: f64) -> f64 {
    let gv = gv.clamp(0.0, 1.0);
    let shaped = GV_FLOOR + (1.0 - GV_FLOOR) * gv;
    1.0 + GV_STRENGTH * (shaped - 0.5) * 2.0
}
