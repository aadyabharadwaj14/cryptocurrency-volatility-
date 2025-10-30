#!/usr/bin/env Rscript
# True joint-MLE GARCHX using rugarch (Single aligned CSV version)
# ---------------------------------------------------------------
# Expects: data/processed/aligned_returns_features.csv
# Columns: timestamp, price, ret, n_nodes, n_edges, total_volume, avg_degree, avg_clustering

suppressPackageStartupMessages({
  req <- c("rugarch", "data.table")
  for (p in req) if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cloud.r-project.org")
  library(rugarch)
  library(data.table)
})

PRO <- "data/processed"
path <- file.path(PRO, "aligned_returns_features.csv")

if (!file.exists(path)) stop("Missing input file: ", path)

# ---------- Load & prepare ----------
dt <- fread(path)
if (!"timestamp" %in% names(dt)) {
  setnames(dt, names(dt)[1], "timestamp")
}
dt[, timestamp := as.IDate(timestamp)]
setorder(dt, timestamp)

# Keep relevant columns
cols_features <- c("n_nodes", "n_edges", "total_volume", "avg_degree", "avg_clustering")
req_cols <- c("timestamp", "ret", cols_features)
missing <- setdiff(req_cols, names(dt))
if (length(missing) > 0) stop("Missing columns: ", paste(missing, collapse = ", "))

# Causal lag: use X_{t-1} to explain volatility at t
for (col in cols_features) {
  dt[, paste0(col, "_l1") := shift(get(col), 1L, type = "lag")]
}

# Drop first NA (from lagging)
dt <- dt[complete.cases(dt)]

# Transform features: log1p + standardize
feat_cols_lag <- paste0(cols_features, "_l1")
X_raw <- as.matrix(dt[, ..feat_cols_lag])
X_log <- log1p(pmax(X_raw, 0))  # ensure nonnegative before log1p
X <- scale(X_log)

# Target series
r <- dt$ret

# ---------- True joint-MLE fits ----------
spec_garch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
  mean.model     = list(armaOrder = c(0,0), include.mean = FALSE),
  distribution.model = "std"
)

spec_garchx <- ugarchspec(
  variance.model = list(
    model = "sGARCH", garchOrder = c(1,1),
    external.regressors = X
  ),
  mean.model     = list(armaOrder = c(0,0), include.mean = FALSE),
  distribution.model = "std"
)

cat("Fitting true GARCH(1,1)...\n")
fit0 <- ugarchfit(spec = spec_garch, data = r, solver = "hybrid", fit.control = list(scale=1))
cat("Fitting true GARCHX...\n")
fitx <- ugarchfit(spec = spec_garchx, data = r, solver = "hybrid", fit.control = list(scale=1))

# In-sample conditional variances
sigma2_garch  <- sigma(fit0)^2
sigma2_garchx <- sigma(fitx)^2
rv            <- r^2

qlike <- function(realized_var, pred_var, eps = 1e-12) {
  rv <- pmax(realized_var, eps)
  pv <- pmax(pred_var, eps)
  mean(rv / pv - log(rv / pv) - 1)
}

ql_g <- qlike(rv, sigma2_garch)
ql_x <- qlike(rv, sigma2_garchx)

cat("\n=== In-sample results (true joint MLE) ===\n")
cat(sprintf("GARCH   QLIKE: %.4f\n", ql_g))
cat(sprintf("GARCHX  QLIKE: %.4f\n", ql_x))
cat("\nParameter summary (GARCHX):\n")
show(fitx)

# ---------- Save outputs ----------
dir.create(PRO, showWarnings = FALSE, recursive = TRUE)

out_var <- data.table(
  timestamp = dt$timestamp,
  realized_var = rv,
  garch_var = as.numeric(sigma2_garch),
  garchx_var = as.numeric(sigma2_garchx)
)
fwrite(out_var, file.path(PRO, "true_garchx_variances.csv"))

summ <- data.table(
  model = c("GARCH", "GARCHX"),
  QLIKE = c(ql_g, ql_x),
  AIC   = c(infocriteria(fit0)["Akaike",1], infocriteria(fitx)["Akaike",1]),
  BIC   = c(infocriteria(fit0)["Bayes",1],  infocriteria(fitx)["Bayes",1])
)
fwrite(summ, file.path(PRO, "true_garchx_in_sample_results.csv"))

cat("\nSaved:\n - data/processed/true_garchx_variances.csv\n - data/processed/true_garchx_in_sample_results.csv\n")
