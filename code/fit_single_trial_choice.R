# Fit nested models to responded human offers. All scaling is learned in training.
suppressPackageStartupMessages(library(lme4))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) stop("Usage: Rscript fit_single_trial_choice.R private_trials.tsv output-directory [--activation-only]")
activation_only <- "--activation-only" %in% args
out <- args[2]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
d <- read.delim(args[1], stringsAsFactors = FALSE)
recovered <- "--recovered" %in% args
stopifnot(if (recovered) nrow(d)>4382 && nrow(d)<=4438 else nrow(d)==4382, length(unique(d$subject)) == 47, !anyNA(d),
          !anyDuplicated(d[c("subject","run","trial")]), all(d$accept %in% 0:1),
          all(d$partner %in% c(-.5,.5)), all(d$offer %in% 1:10))
d$subject <- factor(d$subject)
d$offer_c <- (d$offer - 5.5) / 3
d$order_c <- (d$trial - 36.5) / 36
d$run2 <- as.numeric(d$run == 2L)
d$age <- d$age - .5
subjects <- unique(d[c("subject", "age")])
subjects <- subjects[order(subjects$subject), ]
set.seed(20140203)
subjects$fold <- ave(seq_len(nrow(subjects)), subjects$age, FUN = function(x)
  sample(rep(1:5, length.out=length(x))))
d$fold <- subjects$fold[match(d$subject, subjects$subject)]
base <- paste("accept ~ offer_c * partner * age + I(offer_c^2) + I(offer_c^3)",
              "+ fair + run2 + order_c + ismale + tsnr + fd_mean + (1 + offer_c | subject)")
amp_terms <- function(sites) as.vector(outer(sites,c("_within","_mean"),paste0))
if ("--whole-brain" %in% args) {
  stopifnot(!recovered, nrow(d)==4382L, all(is.finite(d$rt)))
  d$rt_mean <- ave(d$rt, d$subject, FUN=mean)
  d$rt_within <- d$rt - d$rt_mean
  behavior <- update(as.formula(base), . ~ . + rt_mean + rt_within)
  sites <- c("medial_frontal", "precuneus", "visual", "dmn", "frontal_target")
  brain <- amp_terms(sites)
  activation <- update(behavior, as.formula(paste(". ~ . +", paste(brain, collapse=" + "))))
  joint <- update(activation, . ~ . + dmn_within:frontal_target_within)
  forms <- list(behavior=behavior, activation=activation, same_trial=joint)
  comparisons <- list(activation=c("behavior","activation"), same_trial=c("activation","same_trial"))
  age_terms <- paste(paste0("age:",sites,"_within"),collapse=" + ")
  activation_age <- update(activation, as.formula(paste(". ~ . +",age_terms)))
  joint_age_base <- update(activation_age,. ~ . + dmn_within:frontal_target_within)
  forms$activation_age <- activation_age
  forms$joint_age_base <- joint_age_base
  forms$joint_age <- update(joint_age_base,. ~ . + age:dmn_within:frontal_target_within)
  comparisons$activation_age <- c("activation","activation_age")
  comparisons$same_trial_age <- c("joint_age_base","joint_age")
  scale_vars <- c("tsnr", "fd_mean", "rt_mean", "rt_within", brain)
  if ("opposite_run_dmn" %in% names(d)) {
    forms$gppi <- update(activation, . ~ . + opposite_run_dmn * offer_c * partner)
    forms$joint_gppi <- update(joint, . ~ . + opposite_run_dmn * offer_c * partner)
    comparisons$gppi <- c("activation","gppi")
    comparisons$gppi_after_same_trial <- c("same_trial","joint_gppi")
    forms$gppi_age_base <- update(activation_age,. ~ . + opposite_run_dmn * offer_c * partner)
    forms$gppi_age <- update(forms$gppi_age_base,. ~ . + opposite_run_dmn * offer_c * partner * age)
    comparisons$gppi_age <- c("gppi_age_base","gppi_age")
    scale_vars <- c(scale_vars,"opposite_run_dmn")
  }
} else {
  families <- list(dmn_atlas=list(regions=c("vmpfc","insula","dmn","pole"), ppi="dmn_pole"),
                   dmn_focus=list(regions=c("vmpfc","insula","dmn","focus"), ppi="dmn_focus"),
                   ecn_vmpfc=list(regions=c("vmpfc","insula","ecn"), ppi="ecn_vmpfc"))
  if (activation_only || recovered) families <- families["dmn_atlas"]
  forms <- list(behavior=as.formula(base))
  comparisons <- list()
  for (f in names(families)) {
    spec <- families[[f]]
    regions <- amp_terms(spec$regions)
    a <- paste0("activation_",f)
    forms[[a]] <- as.formula(paste(base,"+",paste(regions,collapse=" + ")))
    if (f == "dmn_atlas") comparisons$activation <- c("behavior",a)
    if (!activation_only) {
      forms[[f]] <- update(forms[[a]],paste(". ~ . +",spec$ppi,"* offer_c * partner"))
      comparisons[[f]] <- c(a,f)
    }
  }
  scale_vars <- unique(c("tsnr","fd_mean", unlist(lapply(families,function(x) amp_terms(x$regions))),
                        if (!activation_only) vapply(families,function(x)x$ppi, "")))
}
needed <- unique(unlist(lapply(forms,all.vars)))
missing_vars <- setdiff(needed,names(d))
if (length(missing_vars)) stop("Missing model inputs: ",paste(missing_vars,collapse=", "))
control <- function(optimizer) glmerControl(optimizer=optimizer,
    optCtrl=if (optimizer=="bobyqa") list(maxfun=150000) else list(maxeval=150000))
fit_model <- function(form,train) {
  for (opt in c("nloptwrap","bobyqa")) {
    fit <- glmer(form, data=train, family=binomial(),control=control(opt),nAGQ=1)
    msg <- fit@optinfo$conv$lme4$messages
    # A boundary fit is retained and counted explicitly; unresolved optimizer failures stop publication.
    if (fit@optinfo$conv$opt == 0 && !length(msg[!grepl("boundary.*singular",msg)]) &&
        !length(attr(getME(fit,"X"),"col.dropped"))) return(fit)
  }
  stop("Unresolved convergence or rank problem in ", deparse(form))
}
population_prediction <- function(fit,test) {
  fixed <- predict(fit,newdata=test,re.form=NA,type="link",allow.new.levels=TRUE)
  cov <- as.matrix(VarCorr(fit)$subject)
  sd <- sqrt(pmax(0,cov[1,1]+2*test$offer_c*cov[1,2]+test$offer_c^2*cov[2,2]))
  # Integrate the one-dimensional normal linear predictor. Check every test probability
  # at two integration resolutions, including offers with large random-slope variance.
  previous <- NULL
  for (step in c(.1, .05, .025, .0125)) {
    nodes <- seq(-9,9,by=step)
    weights <- dnorm(nodes) / sum(dnorm(nodes))
    p <- as.numeric(plogis(fixed + outer(sd,nodes)) %*% weights)
    if (!is.null(previous) && max(abs(p-previous)) < 1e-8) return(p)
    previous <- p
  }
  stop("Marginal test probabilities did not stabilize")
}
run_fold <- function(k) {
  train <- if (k==0) d else d[d$fold!=k,]
  test <- if (k==0) NULL else d[d$fold==k,]
  for (name in scale_vars) {
    center <- mean(train[[name]])
    spread <- sd(train[[name]])
    stopifnot(is.finite(spread), spread>0)
    train[[name]] <- (train[[name]]-center)/spread
    if (k>0) test[[name]] <- (test[[name]]-center)/spread
  }
  diagnostics <- list()
  models <- lapply(forms,fit_model,train=train)
  losses <- list()
  for (name in names(models)) {
    fit <- models[[name]]
    diagnostics[[name]] <- data.frame(fold=k, model=name, n=nrow(train),
        participants=length(unique(train$subject)), singular=isSingular(fit), optimizer=fit@optinfo$optimizer,
        loglik=as.numeric(logLik(fit)), parameters=attr(logLik(fit),"df"))
    if (k>0) {
      p <- pmin(1-1e-15,pmax(1e-15,population_prediction(fit,test)))
      losses[[name]] <- data.frame(subject=test$subject,model=name,age=test$age,
         log_loss=-ifelse(test$accept==1L,log(p),log1p(-p)),brier=(test$accept-p)^2)
    }
  }
  tests <- coef_table <- NULL
  if (k==0) {
    tests <- do.call(rbind,lapply(names(comparisons),function(n) {
      pair <- comparisons[[n]]
      small <- logLik(models[[pair[1]]]); large <- logLik(models[[pair[2]]])
      lr <- as.numeric(2*(large-small)); df <- attr(large,"df")-attr(small,"df")
      if (lr < -.01 || df<1) stop("Nested model check failed")
      data.frame(comparison=n, reduced=pair[1], extended=pair[2], likelihood_ratio=lr, df=df,
                 p_asymptotic=pchisq(max(0,lr),df,lower.tail=FALSE))
    }))
    coef_table <- do.call(rbind,lapply(names(models),function(name){
      x <- coef(summary(models[[name]]))
      data.frame(model=name,term=rownames(x), estimate=x[,1],standard_error=x[,2])
    }))
  }
  list(diag=do.call(rbind,diagnostics), loss=do.call(rbind,losses), tests=tests, coefs=coef_table)
}
workers <- min(6L,max(1L,as.integer(Sys.getenv("CHOICE_WORKERS","3"))))
results <- parallel::mclapply(0:5,run_fold,mc.cores=workers,mc.preschedule=FALSE)
failed <- which(vapply(results,function(x)!is.list(x) || is.null(x$diag),logical(1)))
if (length(failed)) stop("Model fits failed in fold(s) ",paste(failed-1L,collapse=", "),
                         ". First error: ",paste(results[[failed[1]]],collapse=" "))
gather <- function(key) do.call(rbind,lapply(results,"[[",key))
write_tsv <- function(x,filename) write.table(x,file.path(out,filename),sep="\t",quote=FALSE,row.names=FALSE)
write_tsv(gather("diag"),"diagnostics.tsv")
write_tsv(gather("tests"),"nested_tests.tsv")
write_tsv(gather("coefs"),"coefficients.tsv")
loss <- gather("loss")
stopifnot(nrow(loss)==nrow(d)*length(forms))
by_id <- aggregate(cbind(log_loss,brier)~subject+age+model,loss,mean)
write_tsv(aggregate(cbind(log_loss,brier)~model,by_id,mean),"heldout_scores.tsv")
set.seed(20141209)
summary <- do.call(rbind,lapply(names(comparisons),function(n) {
  pair <- comparisons[[n]]
  x <- merge(subset(by_id, model==pair[1]),subset(by_id, model==pair[2]),by=c("subject","age"))
  stopifnot(nrow(x)==47)
  delta <- x$log_loss.x - x$log_loss.y
  boot <- replicate(5000,mean(delta[unlist(lapply(split(seq_len(nrow(x)),x$age),function(ix)
    sample(ix,length(ix),replace=TRUE)))]))
  ci <- quantile(boot,c(.025,.975),names=FALSE)
  data.frame(comparison=n, participants=nrow(x), trials=nrow(d),
       log_loss_reduction=mean(delta),conditional_lo=ci[1],conditional_hi=ci[2],
       brier_reduction=mean(x$brier.x-x$brier.y),participants_improved=sum(delta>0))
}))
write_tsv(summary,"heldout_gains.tsv")
if ("--whole-brain" %in% args) {
  group_summary <- do.call(rbind,lapply(names(comparisons),function(n) {
    pair <- comparisons[[n]]
    x <- merge(subset(by_id, model==pair[1]),subset(by_id, model==pair[2]),by=c("subject","age"))
    gain <- split(x$log_loss.x-x$log_loss.y, x$age)
    younger <- gain[["-0.5"]]; older <- gain[["0.5"]]
    stopifnot(length(younger)==25L,length(older)==22L)
    boot <- replicate(5000,mean(sample(older,replace=TRUE))-mean(sample(younger,replace=TRUE)))
    ci <- quantile(boot,c(.025,.975),names=FALSE)
    data.frame(comparison=n, younger_gain=mean(younger), older_gain=mean(older),
               older_minus_younger=mean(older)-mean(younger),conditional_lo=ci[1],conditional_hi=ci[2])
  }))
  write_tsv(group_summary,"heldout_by_age.tsv")
}
writeLines(c(R.version.string,paste("lme4",packageVersion("lme4"))),file.path(out,"software.txt"))
print(summary,row.names=FALSE)
