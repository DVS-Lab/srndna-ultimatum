clear; close all;

%% set up dirs
scriptname = matlab.desktop.editor.getActiveFilename;
[codedir,~,~] = fileparts(scriptname);
cd(codedir);
addpath(codedir)
cd ..
maindir = pwd;
roidir = fullfile(maindir,'derivatives','imaging_plots');

%% load participant information
P = readtable(fullfile(roidir,'participants.tsv'),'FileType','delimitedtext');


%% loop through rois for activation
rois = {'roi-VMPFC', 'roi-VS', 'roi-aINS', 'roi-dACC', 'func-DLPFC'};
for r = 1:length(rois)
    roi = rois{r};
    
    % all of these are the parametric effects of fairness -- i.e., larger
    % responses mean stronger association with the amount offered
    c2 = load(fullfile(roidir,[roi '_type-act_cope-02.txt']));
    c4 = load(fullfile(roidir,[roi '_type-act_cope-04.txt']));
    c6 = load(fullfile(roidir,[roi '_type-act_cope-06.txt']));
    
    % write table for Rita
    brainData = array2table([c2 c4 c6],'VariableNames',{'comp_p','in_p','out_p'});
    outT = [P brainData];
    outfile = fullfile(roidir,['activation_' roi '_modulated.csv']);
    writetable(outT,outfile,'Delimiter',',')
    
end


%% connectivity results
roi = 'func-insula'; % this shows older adults > younger adults for social (in/out) > nonsocial (computer)

% all of these are strictly the effect of partner, independent of offer amount
c1 = load(fullfile(roidir,[roi '_type-nppi-ecn_cope-01.txt']));
c3 = load(fullfile(roidir,[roi '_type-nppi-ecn_cope-03.txt']));
c5 = load(fullfile(roidir,[roi '_type-nppi-ecn_cope-05.txt']));

% write table for Rita
brainData = array2table([c1 c3 c5],'VariableNames',{'comp','in','out'});
outT = [P brainData];
outfile = fullfile(roidir,['conn_ecn-' roi '_unmodulated.csv']);
writetable(outT,outfile,'Delimiter',',')


roi = 'func-dACC'; % this shows older adults > younger adults for social (in/out) > nonsocial (computer)

% all of these are strictly the effect of partner, independent of offer amount
c2 = load(fullfile(roidir,[roi '_type-nppi-dmn_cope-02.txt']));
c4 = load(fullfile(roidir,[roi '_type-nppi-dmn_cope-04.txt']));
c6 = load(fullfile(roidir,[roi '_type-nppi-dmn_cope-06.txt']));

% write table for Rita
brainData = array2table([c2 c4 c6],'VariableNames',{'comp_p','in_p','out_p'});
outT = [P brainData];
outfile = fullfile(roidir,['conn_dmn-' roi '_modulated.csv']);
writetable(outT,outfile,'Delimiter',',')



