/// Classification nonconformity s(x,y) = 1 - p_y
pub fn class_score(prob_of_true: f64) -> f64 {
    let p = prob_of_true.clamp(0.0, 1.0);
    1.0 - p
}

/// Regression nonconformity: absolute residual
pub fn regr_score(y_true: f64, y_pred: f64) -> f64 {
    (y_true - y_pred).abs()
}
