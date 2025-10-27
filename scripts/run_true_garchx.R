#!/usr/bin/env Rscript
# True joint-MLE GARCHX using rugarch
# - Reads your existing CSVs
# - Aligns & lags ETH features (causal)
# - Fits GARCH(1,1) and GARCHX jointly
# - Compares QLIKE (lower = better)
# - Writes results to data/processed/

suppressPackageStartupMessages({
  req <- c("rugarch", "data.table")
  for (p in req) if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cloud.r-project.org")
  library(rugarch)
  library(data.table)
})

PRO <- "data/processed"

# ---------- Helpers ----------
qlike <- function(realized_var, pred_var, eps = 1e-12) {
  rv <- pmax(realized_var, eps)
  pv <- pmax(pred_var, eps)
  mean(rv / pv - log(rv / pv) - 1)
}

# Robust CSV reader that works whether date is a column or the index
read_returns <- function(path) {
  dt <- fread(path)
  # Expect columns: timestamp, price (optional), ret
  if (!"timestamp" %in% names(dt)) {
    # first column is likely the timestamp index
    setnames(dt, names(dt)[1], "timestamp")
  }
  dt[, timestamp := as.IDate(timestamp)]
  dt[order(timestamp)]
}

read_features <- function(path) {
  dt <- fread(path)
  # Expect first col = date (from your Python saver)
  if ("date" %in% names(dt)) {
    setnames(dt, "date", "timestamp")
  } else if (!"timestamp" %in% names(dt)) {
    setnames(dt, names(dt)[1], "timestamp")
  }
  dt[, timestamp := as.IDate(timestamp)]
  dt[order(timestamp)]
}

# ---------- Load & align ----------
ret_path <- file.path(PRO, "BTC_returns_daily.csv")
feat_path <- file.path(PRO, "ETH_graph_features_daily.csv")

if (!file.exists(ret_path)) stop("Missing: ", ret_path)
if (!file.exists(feat_path)) stop("Missing: ", feat_path)

rdt  <- read_returns(ret_path)
xdt  <- read_features(feat_path)

# Keep needed cols
rdt  <- rdt[, .(timestamp, ret)]
xdt  <- xdt[, .(timestamp, n_nodes, n_edges, total_volume, avg_degree, avg_clustering)]

# Causal lag: use X_{t-1} to explain volatility at t
setkey(rdt, timestamp); setkey(xdt, timestamp)
xdt[, `:=`(
  n_nodes_l1       = shift(n_nodes, 1L, type = "lag"),
  n_edges_l1       = shift(n_edges, 1L, type = "lag"),
  total_volume_l1  = shift(total_volume, 1L, type = "lag"),
  avg_degree_l1    = shift(avg_degree, 1L, type = "lag"),
  avg_clustering_l1= shift(avg_clustering, 1L, type = "lag")
)]
# Use only the lagged versions (cleanest)
xlag <- xdt[, .(timestamp, n_nodes_l1, n_edges_l1, total_volume_l1, avg_degree_l1, avg_clustering_l1)]

# Merge and clean
dt <- merge(rdt, xlag, by = "timestamp", all = FALSE)
dt <- dt[complete.cases(dt)]  # drop the first lagged NA and any gaps

# Transform features: log1p + standardize
feat_cols <- c("n_nodes_l1","n_edges_l1","total_volume_l1","avg_degree_l1","avg_clustering_l1")
X_raw <- as.matrix(dt[, ..feat_cols])
X_log <- log1p(pmax(X_raw, 0))   # ensure nonnegative before log1p
X    <- scale(X_log)

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

ql_g <- qlike(rv, sigma2_garch)
ql_x <- qlike(rv, sigma2_garchx)

cat("\n=== In-sample results (true joint MLE) ===\n")
cat(sprintf("GARCH   QLIKE: %.4f\n", ql_g))
cat(sprintf("GARCHX  QLIKE: %.4f\n", ql_x))
cat("\nParameter summary (GARCHX):\n")
show(fitx)

# ---------- Save outputs ----------
dir.create(PRO, showWarnings = FALSE, recursive = TRUE)

# Variances + realized
out_var <- data.table(
  timestamp = dt$timestamp,
  realized_var = rv,
  garch_var = as.numeric(sigma2_garch),
  garchx_var = as.numeric(sigma2_garchx)
)
fwrite(out_var, file.path(PRO, "true_garchx_variances.csv"))

# Summary table
summ <- data.table(
  model = c("GARCH","GARCHX"),
  QLIKE = c(ql_g, ql_x),
  AIC   = c(infocriteria(fit0)["Akaike",1], infocriteria(fitx)["Akaike",1]),
  BIC   = c(infocriteria(fit0)["Bayes",1],  infocriteria(fitx)["Bayes",1])
)
fwrite(summ, file.path(PRO, "true_garchx_in_sample_results.csv"))

cat("\nSaved:\n - data/processed/true_garchx_variances.csv\n - data/processed/true_garchx_in_sample_results.csv\n")
