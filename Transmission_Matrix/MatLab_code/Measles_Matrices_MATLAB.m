%% Alberta measles contact and transmission matrices Calculation 
% We use 2025 population data
%  procedure; population.csv in current working directory.
clear; clc;
T = readtable('population.csv','TextType','string');
assert(height(T)==20 && all(T.Year==2025), ...
    'Expected the 20-row Alberta 2025 data.');
assert(all(T.Geography=="AB") && all(T.Sex=="BOTH"), ...
    'Expected Alberta, both sexes.');
expectedAges = ["00";"01to04";"05to09";"10to14";"15to19"; ...
    "20to24";"25to29";"30to34";"35to39";"40to44"; ...
    "45to49";"50to54";"55to59";"60to64";"65to69"; ...
    "70to74";"75to79";"80to84";"85to89";"90+"];
assert(isequal(string(T.Age), expectedAges),'Unexpected age ordering.');
% 20 CSV rows -> 15 POLYMOD categories (0-4, 5-9, ..., 65-69, 70+).
population = [sum(T.Population(1:2)); T.Population(3:15); ...
              sum(T.Population(16:20))];

% Original 15-by-15 POLYMOD-GB matrix from the class:
contact_gb = [
1.92 0.65 0.41 0.24 0.46 0.73 0.67 0.83 0.24 0.22 0.36 0.20 0.20 0.26 0.13;
0.95 6.64 1.09 0.73 0.61 0.75 0.95 1.39 0.90 0.16 0.30 0.22 0.50 0.48 0.20;
0.48 1.31 6.85 1.52 0.27 0.31 0.48 0.76 1.00 0.69 0.32 0.44 0.27 0.41 0.33;
0.33 0.34 1.03 6.71 1.58 0.73 0.42 0.56 0.85 1.16 0.70 0.30 0.20 0.48 0.63;
0.45 0.30 0.22 0.93 2.59 1.49 0.75 0.63 0.77 0.87 0.88 0.61 0.53 0.37 0.33;
0.79 0.66 0.44 0.74 1.29 1.83 0.97 0.71 0.74 0.85 0.88 0.87 0.67 0.74 0.33;
0.97 1.07 0.62 0.50 0.88 1.19 1.67 0.89 1.02 0.91 0.92 0.61 0.76 0.63 0.27;
1.02 0.98 1.26 1.09 0.76 0.95 1.53 1.50 1.32 1.09 0.83 0.69 1.02 0.96 0.20;
0.55 1.00 1.14 0.94 0.73 0.88 0.82 1.23 1.35 1.27 0.89 0.67 0.94 0.81 0.80;
0.29 0.54 0.57 0.77 0.97 0.93 0.57 0.80 1.32 1.87 0.61 0.80 0.61 0.59 0.57;
0.33 0.38 0.40 0.41 0.44 0.85 0.60 0.61 0.71 0.95 0.74 1.06 0.59 0.56 0.57;
0.31 0.21 0.25 0.33 0.39 0.53 0.68 0.53 0.55 0.51 0.82 1.17 0.85 0.85 0.33;
0.26 0.25 0.19 0.24 0.19 0.34 0.40 0.39 0.47 0.55 0.41 0.78 0.65 0.85 0.57;
0.09 0.11 0.12 0.20 0.19 0.22 0.13 0.30 0.23 0.13 0.21 0.28 0.36 0.70 0.60;
0.14 0.15 0.21 0.10 0.24 0.17 0.15 0.41 0.50 0.71 0.53 0.76 0.47 0.74 1.47
];

num_groups = 2;
groups = {1, 2:15};    % group 1: 0-4; group 2: 5+
matrix_a = zeros(15,num_groups);
for i = 1:15
    for j = 1:num_groups
        matrix_a(i,j) = sum(contact_gb(i,groups{j}));
    end
end

contact = zeros(num_groups,num_groups);
for i = 1:num_groups
    for j = i:num_groups
        c_ij = dot(matrix_a(groups{j},i),population(groups{j}));
        c_ji = dot(matrix_a(groups{i},j),population(groups{i}));
        average = (c_ij+c_ji)/2;
        contact(i,j) = average/sum(population(groups{i}));
        contact(j,i) = average/sum(population(groups{j}));
    end
end
N1 = population(1);
N2 = sum(population(2:15));
W12 = dot(matrix_a(groups{1},2),population(groups{1}));
W21 = dot(matrix_a(groups{2},1),population(groups{2}));
Wbar = (W12+W21)/2;
C = contact;

beta1 = 0.9; beta2 = 0.9;  % assumed high-transmission scenario
beta = [beta1,0;0,beta2];
beta_matrix = beta*contact;  % preserve class matrix multiplication

fprintf('N1 = %.6f; N2 = %.6f\n',N1,N2);
fprintf('W12 = %.9f; W21 = %.9f; Wbar = %.9f\n',W12,W21,Wbar);
fprintf('C i.e Contact matrix =\n'); disp(C);
fprintf('B i.e Transmissin matrix = diag(beta1,beta2)*C =\n'); disp(beta_matrix);
fprintf('Reciprocity error = %.12g\n',N1*C(1,2)-N2*C(2,1));


