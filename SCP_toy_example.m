
close all
clear
clc

%% Libs and imports

addpath(genpath("../MATLAB_tools/casadi"))

% Figures
fm = FigureManager('exportDir', './figures', ...
                'visible', true, ...
                'export_figs', false, ...
                'saveFig', false);

%% Problem definition and simulation

% Simulation
dt_sim = 1e-2; % s
t_f = 20;      % s
sim_time = 0:dt_sim:t_f;
X_k = [0; 0; 0; 0];
X_kp1 = X_k;
U_k = [0; 0];
% Target and obstacles
final_state = [10;10;0;0];
obs_position    = [5;5];
obs_radius      = 3;

% Control
dt_ctrl = 0.1;
max_accel = 10; % m/s2
time_since_last_ctrl = dt_ctrl;
updated_ctrls = 0;
applied_ctrls = interp1([0;t_f], zeros(2, 2), sim_time, 'nearest', 0);

% SCP
SCP_param = struct();
SCP_param.dt_SCP = dt_ctrl;
SCP_param.dt_sim = dt_sim;
SCP_param.N = 30;
SCP_param.z_k = [repmat(X_k, SCP_param.N+1, 1); repmat(U_k, SCP_param.N, 1)];
SCP_param.nx = 4;
SCP_param.nu = 2;
SCP_param.nz = SCP_param.nx*(SCP_param.N+1) + SCP_param.nu*SCP_param.N;
SCP_param.w_Q_pos = 0;
SCP_param.w_Q_vel = 0;
SCP_param.w_R = 0.01;
SCP_param.wf_Q_pos = 0.1;
SCP_param.wf_Q_vel = 0.01;
SCP_param.Q  = [SCP_param.w_Q_pos 0 0 0; 0 SCP_param.w_Q_pos 0 0; 0 0 SCP_param.w_Q_vel 0; 0 0 0 SCP_param.w_Q_vel];
SCP_param.Qf = [SCP_param.wf_Q_pos 0 0 0; 0 SCP_param.wf_Q_pos 0 0; 0 0 SCP_param.wf_Q_vel 0; 0 0 0 SCP_param.wf_Q_vel];
SCP_param.R  = [SCP_param.w_R 0; 0 SCP_param.w_R];
SCP_param.final_state = final_state;
SCP_param.obs_position    = obs_position;
SCP_param.obs_radius      = obs_radius;
SCP_param.max_accel       = max_accel;
SCP_param.rho           = 1e3;
SCP_param.trust_region  = 1;
SCP_param.alpha         = 1;
SCP_param.slack_max     = 1;

% Compute useful operators for the QP formulation
get_all_operators = initialize_SCP_solver(SCP_param);

% Plots
output_table = initialize_output_table(numel(sim_time));

%% Simulation loop
for i_time = 1:numel(sim_time)

    current_time = sim_time(i_time);

    [ctrl_time, ctrl_values, updated_ctrls, SCP_param] = get_ctrl_values(current_time, X_k, time_since_last_ctrl, SCP_param, get_all_operators);
    if updated_ctrls
        time_since_last_ctrl = dt_sim;
        applied_ctrls = interp1(ctrl_time, ctrl_values', sim_time, 'nearest', 0)';
    end

    U_k = applied_ctrls(:,i_time);

    X_kp1 = apply_dynamics_and_control(X_k, U_k, dt_sim, dt_sim);

    output_table(i_time,:) = num2cell([current_time, X_k', U_k', updated_ctrls]);

    X_k = X_kp1;
    time_since_last_ctrl = time_since_last_ctrl + dt_sim;
end

% Plots
fm.newFigure('overview');
plot_results(output_table)

fm.newFigure('trace');
plot_trace(output_table, SCP_param)

%% Dynamics
function X_kp1 = apply_dynamics_and_control(X_k, U_k, dt, subsample_dt)
    % X_k = [p_x; p_y; v_x; v_y]
    % U_k = [a_x; a_y]

    X_kp1 = X_k;

    if subsample_dt > 0 && subsample_dt < dt
        N_subsamples = round(dt / subsample_dt);
        h = subsample_dt;
    else
        N_subsamples = 1;
        h = dt;
    end

    for i = 1:N_subsamples
        X_next = X_kp1;

        X_next(1) = X_kp1(1) + h * X_kp1(3);
        X_next(2) = X_kp1(2) + h * X_kp1(4);
        X_next(3) = X_kp1(3) + h * U_k(1);
        X_next(4) = X_kp1(4) + h * U_k(2);

        X_kp1 = X_next;
    end
end

%% SCP control

function [ctrl_time, ctrl_values, updated_ctrls, SCP_param] = get_ctrl_values(current_time, current_states, time_since_last_ctrl, SCP_param, get_all_operators)
    % ctrl time : [t0;t1;...;t_N]
    % ctrl_values : [[ux_0, uy_0];[ux_1, uy_1];...;[ux_N, uy_N]]
    % updated_ctrls : if controls have been updated this timestep

    N = SCP_param.N;

    if (time_since_last_ctrl >= SCP_param.dt_SCP)
        
        z_k = update_SCP_z_vector(SCP_param, current_states);

        [sz, scales] = build_Sz_from_z(z_k, SCP_param);

        [~, gJ, HJ, F, JF, C, JC] = get_all_operators(z_k, current_states);

        gJ = full(gJ);
        HJ = full(HJ);
        F  = full(F);
        JF = full(JF);
        C  = full(C);
        JC = full(JC);

        [gJ_n, HJ_n, F_n, JF_n, C_n, JC_n] = normalize_operators(gJ, HJ, F, JF, C, JC, sz, scales, N);

        clear gJ HJ F JF C JC

        % Formulate the QP
        nz = SCP_param.nz;
        nu = SCP_param.nu;
        nx = SCP_param.nx;
        N = SCP_param.N;
        nF = length(F_n);
        nC = length(C_n);

        rho      = SCP_param.rho;
        trust    = SCP_param.trust_region;   % e.g. 0.2 or 0.5 in normalized units
        slackMax = SCP_param.slack_max;      % e.g. 1e-2 to 1

        % --- QP cost ---
        % quadprog solves: 0.5*y'*H_qp*y + f_qp'*y
        H_qp = blkdiag(HJ_n, 2*rho*speye(nF));
        f_qp = [gJ_n; zeros(nF,1)];

        % --- Equality constraints: normalized dynamics ---
        % F + JF*delta_z_norm = s
        % => JF*delta_z_norm - s = -F
        Aeq = [JF_n, -speye(nF)];
        beq = -F_n;

        % --- Inequality constraints ---
        % C + JC*delta_z_norm <= 0
        % => JC*delta_z_norm <= -C
        Aineq = [JC_n, sparse(nC,nF)];
        bineq = -C_n;
    
        % --- Bounds: trust region + slack bounds ---
        lb = [-trust*ones(nz,1); -slackMax*ones(nF,1)];
        ub = [ trust*ones(nz,1);  slackMax*ones(nF,1)];
    
        % --- Solve ---
        opts = optimoptions('quadprog', ...
            'Algorithm','interior-point-convex', ...
            'Display','off');
    
        [y, fval, exitflag, output] = quadprog( ...
            H_qp, f_qp, ...
            Aineq, bineq, ...
            Aeq, beq, ...
            lb, ub, ...
            [], opts);
    
        % --- Extract solution ---
        delta_z_norm = y(1:nz);
%         s_dyn        = y(nz+1:end);
    
        % Convert normalized correction back to physical units
        delta_z = sz .* delta_z_norm;
        
        % SCP update
        z_kp1 = z_k + SCP_param.alpha * delta_z;
        SCP_param.z_k = z_kp1;

        updated_ctrls = 1;
        ctrl_time = current_time:SCP_param.dt_SCP:current_time+SCP_param.dt_SCP * (SCP_param.N-1);
        U_kp1 = z_kp1(nx*(N+1)+1:end);
        ctrl_values = reshape(U_kp1, nu, N);
    else
        updated_ctrls = 0;
        ctrl_time = [current_time; current_time+SCP_param.dt_SCP];
        ctrl_values = [[0;0], [0;0]];
    end
end

function z_k = update_SCP_z_vector(SCP_param, current_states)
    % X_k = [p_x; p_y; v_x; v_y] => nx = 4
    % applied_ctrls = [a_x; a_y] => nu = 2
    % N+1 * X + N * U => Z of size nx(N+1) + nu*N = 6*N+4 here

    N = SCP_param.N;
    nu = SCP_param.nu;
    nx = SCP_param.nx;

    z_k1 = SCP_param.z_k;
    all_X_k1 = z_k1(1:(N+1)*nx);
    all_U_k1 = z_k1((N+1)*nx+1:end);

    % Build U_k
    last_U = all_U_k1(end-(nu-1):end);
    all_U_k = [all_U_k1(nu+1:end); last_U]; % Remove first controls, repeat last

    % Build X_k
    x_0 = current_states;
    last_X = apply_dynamics_and_control(all_X_k1(end-(nx-1):end), last_U, SCP_param.dt_SCP, SCP_param.dt_sim);
    all_X_k = [x_0; all_X_k1(2*nx+1:end); last_X]; % Replace first state with current measurement, add last as if controls were applied

    % Build z_k
    z_k = [all_X_k; all_U_k];

    assert(all(size(z_k) == [nx*(N+1) + nu*N, 1]), "constructed z_k is of wrong size")
end

function J = build_cost(z, SCP_param)
    % In cost : integral of the error on position, final error on velocity,
    % integral of control inputs, with weights (Q and R respectively)
    % J of size (1, 1)
    
    nu = SCP_param.nu;
    nx = SCP_param.nx;
    N  = SCP_param.N;
    
    X = reshape(z(1:nx*(N+1)), nx, N+1);
    U = reshape(z(nx*(N+1)+1:end), nu, N);

    P = X(1:2, :);

%     J_pos = sum((P - [SCP_param.final_state(1);SCP_param.final_state(2)]).^2 * SCP_param.wf_Q_pos, [1 2]);
    J_pos = sum(sum((P - [SCP_param.final_state(1);SCP_param.final_state(2)]).^2 * SCP_param.wf_Q_pos));
    J_vel = (X(3,end) - SCP_param.final_state(3))^2 * SCP_param.wf_Q_vel + (X(4,end) - SCP_param.final_state(4))^2 * SCP_param.wf_Q_vel;
    J_ctrl = sum(sum(U.^2 * SCP_param.w_R));

    J = J_pos + J_vel + J_ctrl;
end

function F = build_eq_constraints(z, current_states, SCP_param)
    % F of length 4N+4 (4N for dynamics, 4 for initial position)
    nu = SCP_param.nu;
    nx = SCP_param.nx;
    N  = SCP_param.N;

    % Initialize F
%     F = zeros(nx*(N+1), 1);
    F = [];
    
    X = reshape(z(1:nx*(N+1)), nx, N+1);
    U = reshape(z(nx*(N+1)+1:end), nu, N);

    % Initial state constraint
    F = [F; X(:,1) - current_states];

    % Dynamics constraints
    for k = 1:N
        % Compute dynamics
%         F(1+k*nx:(k+1)*nx) = apply_dynamics_and_control(X(:, k), U(:, k), SCP_param.dt_SCP, SCP_param.dt_sim) - X(:, k+1);
        F = [F; apply_dynamics_and_control(X(:, k), U(:, k), SCP_param.dt_SCP, SCP_param.dt_sim) - X(:, k+1)];
    end
end

function C = build_ineq_constraints(z, SCP_param)
    % F of length 5N+1 (4N for acceleration bounds, N+1 for obstacle)
    nu = SCP_param.nu;
    nx = SCP_param.nx;
    N  = SCP_param.N;

    % Initialize C
%     C = zeros(2*nu*N + N+1, 1);
    C = [];

    X = reshape(z(1:nx*(N+1)), nx, N+1);
    U = z(nx*(N+1)+1:end);

    % Acceleration saturations
%     C(1:nu*N)        =   U - SCP_param.max_accel;
%     C(nu*N+1:2*nu*N) = - U - SCP_param.max_accel;
    C = [C;   U - SCP_param.max_accel];
    C = [C; - U - SCP_param.max_accel];

    % Obstacle avoidance
    P = X(1:2, :);
%     C(2*nu*N+1:end) = SCP_param.obs_radius^2 - sum((P - SCP_param.obs_position)^2, 1)';
    C = [C; SCP_param.obs_radius^2 - sum((P - SCP_param.obs_position).^2)'];
    
end

function get_all_operators = initialize_SCP_solver(SCP_param)
    % Build CasADi symbolic problem

    import casadi.*

    nx = SCP_param.nx;
    nz = SCP_param.nz;

    % Main decision vector:
    % z = [x0; ...; xN; u0; ...; uN-1]
    z_sym = SX.sym('z', nz, 1);

    % Numeric parameters that change during simulation/MPC
    x0_sym = SX.sym('x0', nx, 1);      % measured current state

    % Build symbolic expressions
    J_sym = build_cost(z_sym, SCP_param);
    F_sym = build_eq_constraints(z_sym, x0_sym, SCP_param);
    C_sym = build_ineq_constraints(z_sym, SCP_param);

    % Derivative operators
    gJ_sym = gradient(J_sym, z_sym);   % dJ/dz
    HJ_sym = hessian(J_sym, z_sym);    % d2J/dz2
    JF_sym = jacobian(F_sym, z_sym);   % dF/dz
    JC_sym = jacobian(C_sym, z_sym);   % dC/dz

    % Callable CasADi evaluator
    get_all_operators = Function( ...
        'eval_problem', ...
        {z_sym, x0_sym}, ...
        {J_sym, gJ_sym, HJ_sym, F_sym, JF_sym, C_sym, JC_sym});
end

function [Sz, scales] = build_Sz_from_z(z, SCP_param)

    nx = SCP_param.nx;
    nu = SCP_param.nu;
    N  = SCP_param.N;

    X = reshape(z(1:nx*(N+1)), nx, N+1);
    U = reshape(z(nx*(N+1)+1:end), nu, N);

    Pos = X(1:2,:);
    Vel = X(3:4,:);

    L = max([1, max(abs(Pos),[],'all'), norm(SCP_param.final_state(1:2)-X(1:2,1))]);
    V = max([1, max(abs(Vel),[],'all')]);
    A = max([1, max(abs(U),[],'all'), SCP_param.max_accel]);

    sx_one = [L; L; V; V];
    su_one = [A; A];

    Sz = [repmat(sx_one, N+1, 1); repmat(su_one, N, 1)];

    scales.L = L;
    scales.V = V;
    scales.A = A;
end

function [gJ_n, HJ_n, F_n, JF_n, C_n, JC_n] = normalize_operators( ...
    gJ, HJ, F, JF, C, JC, sz, scales, N)

    % Diagonal scaling matrices
    Sz = diag(sz);

    % Cost operators
    gJ_n = Sz.' * gJ;
    HJ_n = Sz.' * HJ * Sz;

    % Equality constraints F
    sx = [scales.L; scales.L; scales.V; scales.V];
    sF = repmat(sx, N+1, 1);

    SF_inv = diag(1 ./ sF);

    F_n  = SF_inv * F;
    JF_n = SF_inv * JF * Sz;

    % Inequality constraints C
    sC_acc = scales.A   * ones(4*N, 1);
    sC_obs = ones(N+1, 1);
    sC = [sC_acc; sC_obs];

    SC_inv = diag(1 ./ sC);

    C_n  = SC_inv * C;
    JC_n = SC_inv * JC * Sz;

    % No scaling : do not try this at home
%     gJ_n = gJ;
%     HJ_n = HJ;
%     F_n = F;
%     JF_n = JF;
%     C_n = C;
%     JC_n = JC;
end

%% Functions : Utils

function table = initialize_output_table(N)

    table = struct();
    table.time = zeros(N, 1);
    table.p_x = zeros(N, 1);
    table.p_y = zeros(N, 1);
    table.v_x = zeros(N, 1);
    table.v_y = zeros(N, 1);
    table.u_x = zeros(N, 1);
    table.u_y = zeros(N, 1);
    table.upd_ctrl = zeros(N, 1);

    table = struct2table(table);
end

function plot_results(output_table)
    % Auto-generated by plotMenu
    tiled = tiledlayout(2, 2);
    
    ax = nexttile(tiled, 1); hold(ax,'on');
    hLine_1_1 = plot(ax, output_table.time, output_table.u_x, 'LineWidth', 1.5, 'LineStyle', '-', 'Color', [0 0.447 0.741], 'DisplayName', 'u_x');
    hLine_1_2 = plot(ax, output_table.time, output_table.u_y, 'LineWidth', 1.5, 'LineStyle', '-', 'Color', [0.85 0.325 0.098], 'DisplayName', 'u_y');
    hold(ax,'off');
    ylim(ax, 'padded');
    title(ax, 'acceleration (inputs)');
    xlabel(ax, 'time (s)');
    ylabel(ax, 'acc (m/s2)');
    legend(ax, [hLine_1_1 hLine_1_2], {'u_x','u_y'}, 'Location', 'best');
    grid(ax,'on');
    
    ax = nexttile(tiled, 2); hold(ax,'on');
    hLine_2_1 = plot(ax, output_table.time, output_table.v_x, 'LineWidth', 1.5, 'LineStyle', '-', 'Color', [0 0.447 0.741], 'DisplayName', 'v_x');
    hLine_2_2 = plot(ax, output_table.time, output_table.v_y, 'LineWidth', 1.5, 'LineStyle', '-', 'Color', [0.85 0.325 0.098], 'DisplayName', 'v_y');
    hold(ax,'off');
    ylim(ax, 'padded');
    title(ax, 'speed');
    xlabel(ax, 'time (s)');
    ylabel(ax, 'vel (m/s)');
    legend(ax, [hLine_2_1 hLine_2_2], {'v_x','v_y'}, 'Location', 'best');
    grid(ax,'on');
    
    ax = nexttile(tiled, 3); hold(ax,'on');
    hLine_3_1 = plot(ax, output_table.time, output_table.p_x, 'LineWidth', 1.5, 'LineStyle', '-', 'Color', [0 0.447 0.741], 'DisplayName', 'p_x');
    hLine_3_2 = plot(ax, output_table.time, output_table.p_y, 'LineWidth', 1.5, 'LineStyle', '-', 'Color', [0.85 0.325 0.098], 'DisplayName', 'p_y');
    hold(ax,'off');
    ylim(ax, 'padded');
    title(ax, 'position');
    xlabel(ax, 'time (s)');
    ylabel(ax, 'pos (m)');
    legend(ax, [hLine_3_1 hLine_3_2], {'p_x','p_y'}, 'Location', 'best');
    grid(ax,'on');
    
    ax = nexttile(tiled, 4); hold(ax,'on');
    hLine_4_1 = plot(ax, output_table.time, output_table.upd_ctrl, 'LineWidth', 1.5, 'LineStyle', '-', 'Color', [0 0.447 0.741], 'DisplayName', 'upd_ctrl');
    hold(ax,'off');
    ylim(ax, 'padded');
    title(ax, 'control input times');
    xlabel(ax, 'time (s)');
    ylabel(ax, '(-)');
    legend(ax, [hLine_4_1], {'upd_ctrl'}, 'Location', 'best');
    grid(ax,'on');
end


function plot_trace(output_table, SCP_param)

    % Auto-generated by plotMenu
    tiled = tiledlayout(1, 1);
    
    ax = nexttile(tiled, 1); hold(ax,'on');

    x = SCP_param.obs_position(1); y = SCP_param.obs_position(2); r = SCP_param.obs_radius;
    fplot(@(t) r*sin(t)+x, @(t) r*cos(t)+y, 'Color', 'r');

    hLine_1_1 = plot(ax, output_table.p_x, output_table.p_y, 'LineWidth', 1.5, 'LineStyle', '-', 'Color', [0 0.447 0.741], 'DisplayName', 'Position');
    hold(ax,'off');
    ylim(ax, 'padded');
    title(ax, 'position plot');
    xlabel(ax, 'x (m)');
    ylabel(ax, 'y (m)');
    axis(ax,'equal');
    grid(ax,'on');


end
