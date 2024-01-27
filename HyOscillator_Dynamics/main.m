%% Compute solutions to the oscillator with impacts

%--------------------------------------------------------------------------
% Matlab M-file Project: A Data-Driven Approach for Certifying Asymptotic 
% Stability and Cost Evaluation for Hybrid Systems @  Hybrid Systems Laboratory (HSL)

% Filename: main.m
%--------------------------------------------------------------------------
% Author: Carlos A. Montenegro G.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%   Make sure to install HyEQ Toolbox (Beta) v3.0.0.22 from
%   https://www.mathworks.com/matlabcentral/fileexchange/41372-hybrid-equations-toolbox
%   Copyright @ Hybrid Systems Laboratory (HSL),

clear all; clc;

osc_system = OscillatorImpacts();
x0 = [0.075, 1.1]; % Must lie inside \overline{C} \cap \calU
tspan = [0, 20];
jspan = [0, 30];
config = HybridSolverConfig('refine', 32);
sol = osc_system.solve(x0, tspan, jspan, config);

% Prepare workspace to export
x = sol.x;
ts = sol.t;
jts = sol.jump_times;

% Export workspace
save('../HyArc')