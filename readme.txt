# MABFDR
Code for the MAB-FDR framework introduced in "A framework for Multi-A(rmed)/B(andit) Testing with Online FDR Control", NIPS 2017  arXiv preprint available at https://arxiv.org/abs/1706.05378

--------------- For reconstructing the plots ------------------------

0. pip install sortedcontainers
If you use a remote, uncomment ‘agg’

1. Run experiments
For Gaussian:
python main_onlocal.py --dist-style 1
For Bernoulli:
python main_onlocal.py --dist-style 0 
For Bernoulli New Yorker:
python main_onlocal.py --ny-data 1

2. Generate power plots
For Gaussians:
python plot_main.py --dist-style 1
Plots in the paper are found in plots/BDRvsTT_D1.... and plots/SPSvsNA_D1...

For Bernoulli:
python plot_main.py --dist-style 0
Plots in the paper are found in plots/BDRvsTT_D0_MS2.... and plots/SPSvsNA_D0_MS2...

For Bernoulli from New Yorker:
python plot_main.py --dist-style 0 --ny-data 1
Plots in the paper are found in plots/BDRvsTT_D0_MS0.... and plots/SPSvsNA_D0_MS0...

3. Generate FDP and mFDR plots for Gaussians
python main_onlocal.py --dist-style 1 --power-plot 0
python plot_main.py --dist-style 1 --power-plot 0

Plots in the paper are found in plots/FDPvsHYP... and plots/mFDRvsPi...


----------------- Brief overview over code structure -------------------- 

********** High level scripts *********************************
main_onlocal.py:	runs a for loop which calls single experiments using run_single() in exp_new.py and saves 
plot_main.py:		produces and saves power plots

********** Functional scripts that are used above *************
generate_mu.py:	 	generates experimental settings (means and which hypotheses are true nulls and alternatives)
parse_mu.py:	 	loads mu vectors from file or calls generate_mu.py if the settings do not exist yet
parse_new_yorker.py:	generates experiment for New Yorker experiment with realistic distribution of means

exp_new.py:		Runs a sequence of experiments using online FDR
rowexp_new.py:		class for one experiment using MAB
plot_results.py:	Produces power and sample plots

exp_new_punif.py:	Runs a sequence of experiments using online FDR, drawing uniform p-value for null values
plot_results_punif.py:	Produces FDP and mFDR plots

Online FDR procedures can be found in the folder onlineFDR_proc

********** Parameters which are used by all of the modules **********
dist_style:
Distribution type
0: Bernoulli
1: Gaussian

mu_style:
Different ways you can choose your means to be distributed
1: one peak, rest same low (with sigma noise)
2: one high, uniform down
3: some same around highest (epsilon), rest same low.
4: some same around highest, rest uniform down

hyp_style:
Different ways you can simulate how true nulls and alternatives are distributed
0: uniform across num_hyp
1: many alt at beginning - lin prob
2: many alt at end - lin prob
3: many alt at beginning - step down
4: many alt at beginning - step up

FDRrange (online FDR procedure)
0: LORD++ ( we use the ++ version, described in the paper "Online control of the false discovery rate with decaying memory" available at https://arxiv.org/abs/1710.00499, for the LORD procedure described in our paper )
1: LORD
3: no FDR, plain alpha
4: alpha invest (Foster & Stine '07 )
5: Bonferroni

proc_list (MAB combined with different online multiple testing procedures)
0: MAB-LORD (corrected with LORD++)
3: MAB-IND (uncorrected)
5: MAB-Bonf 

alg_list ( sampling procedure per experiment )
0: LUCB best-arm MAB algorithm ( Kalyanakrishnan et al. '12 )
1: Uniform sampling

------------------------- Some notes ---------------------------
- If you run the code on your local computer, you may comment out the lines in import section of the plotting related code containing 'agg'
- Although AlphaInvest was implemented, it is not plotted in the graphs
- Epsilon-modified LUCB is not currently implemented
- The number of runs that are averaged over are aggregated. Thus, if you run the exp multiple times, you'll average over more samples
- For parallelization (of the e.g. New Yorker experiments) you may write a wrapper and use ipyparallel
