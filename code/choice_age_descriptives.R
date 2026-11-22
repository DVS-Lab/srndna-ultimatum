# Additional fixed-effect contrasts and random-slope diagnostic for the v2 choice model.
# Run with the same table, R, and fitting helpers as the hierarchical comparisons.
choice_descriptives <- function(models, forms, train, directory, fit_model) {
  b <- models$behavior
  ri <- fit_model(update(forms$behavior,. ~ . - (1+offer_c|subject) + (1|subject)),train)
  diag <- data.frame(model=c("random_intercept","random_intercept_and_offer_slope"),
                     loglik=c(as.numeric(logLik(ri)),as.numeric(logLik(b))),AIC=c(AIC(ri),AIC(b)))
  write.table(diag,file.path(directory,"random_effects.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
  m <- models$joint_age
  beta <- fixef(m)
  covariance <- as.matrix(vcov(m))
  rows <- lapply(c(-.5,.5),function(a) {
    weight <- setNames(numeric(length(beta)),names(beta))
    weight["dmn_within:frontal_target_within"] <- 1
    weight["age:dmn_within:frontal_target_within"] <- a
    est <- sum(weight*beta)
    se <- sqrt(drop(t(weight)%*%covariance%*%weight))
    data.frame(group=if(a<0)"younger" else "older", estimate=est, se=se, lo=est-1.96*se, hi=est+1.96*se)
  })
  write.table(do.call(rbind,rows),file.path(directory,"same_trial_by_age.tsv"),sep="\t",quote=FALSE,row.names=FALSE)
}
