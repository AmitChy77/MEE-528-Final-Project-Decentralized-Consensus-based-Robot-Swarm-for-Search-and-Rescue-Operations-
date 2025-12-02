function Model_3()
% Model 3: NO location sharing, WITH workload balancing
% ANIMATED VERSION with extra time after all targets are found

%% ---------------- Parameters ----------------
box = [0 20; 0 20];
N   = 8;                  % robots
M   = 6;                  % targets
T   = 1500;               % maximum iterations
dt  = 0.12;               % timestep

% ---- extra simulation after all targets found ----
extra_time_after_all_found  = 20;                 % seconds
extra_steps_after_all_found = ceil(extra_time_after_all_found / dt);

% ---- control gains ----
k_lloyd     = 0.60;       % centroid pull
k_frontier  = 1.50;       % frontier (unknown) pull
k_consensus = 0.50;       % workload balancing - ENABLED
k_repulse   = 0.30;       % inter-robot spacing
k_wall      = 0.80;       % inward near walls
bw_wall     = 0.60;       % wall thickness (tanh scale)
k_center    = 0.02;       % tiny centering
eps_dither  = 0.06;       % base noise scale

% ---- frontier parameter ----
tau_s = 0.15;             % frontier weight vs mean workload
tau_c = 0.20;             % consensus smoothness - ENABLED

% ---- repulsion parameter ----
d0    = 0.80;             % repulsion length

% ---- detection (NO sharing) ----
sense_R       = 0.95;     % detection radius

% ---- anti-stall extras ----
u0    = 0.12;             % small-force threshold for adaptive noise
gamma = 0.03;             % transition width
k_sweep = 0.04;           % tiny rotating drift magnitude
omega   = 0.10;           % rad/step (drift angular speed)

% ---- grid / density ----
Gx = 160; Gy = 96;
[xg, yg] = meshgrid(linspace(box(1,1), box(1,2), Gx), ...
                    linspace(box(2,1), box(2,2), Gy));
dx = (box(1,2)-box(1,1))/(Gx-1); 
dy = (box(2,2)-box(2,1))/(Gy-1); 
dA = dx*dy;
rho = ones(size(xg));
K_smooth = [0 1 0; 1 4 1; 0 1 0]/8;

% ---- init robots/targets ----
rng(60); % SAME SEED FOR ALL MODELS
P = [rand(N,1)*diff(box(1,:))+box(1,1), ...
     rand(N,1)*diff(box(2,:))+box(2,1)];
Targets = [rand(M,1)*(diff(box(1,:))-1.6)+box(1,1)+0.8, ...
           rand(M,1)*(diff(box(2,:))-1.6)+box(2,1)+0.8];
found = false(M,1); 
found_time = nan(M,1);
FoundCoords = nan(0,2);

% ---- viz/logs ----
trail_colors   = lines(N);
Ptrail         = nan(T,N,2); 
Jhist          = nan(T,1);
num_found_hist = zeros(T,1);

Bx = [box(1,1) box(1,2) box(1,2) box(1,1)];
By = [box(2,1) box(2,1) box(2,2) box(2,2)];
Bpoly  = polyshape(Bx, By);
center = mean(box,2)';

% ---- Setup figure ----
figName = 'Model 3: Workload Balancing Only (No Location Sharing) - Animated';
f = figure('Color','w','Name',figName, 'Position', [100 100 1400 600]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

ax1 = nexttile(1); 
axis(ax1,'equal'); grid(ax1,'on');
xlim(ax1, box(1,:)); ylim(ax1, box(2,:));
xlabel(ax1,'x'); ylabel(ax1,'y'); 
title(ax1,'Coverage, trajectories, targets');

ax2 = nexttile(2); 
hold(ax2,'on'); grid(ax2,'on');
title(ax2,'(Left) J(t)  |  (Right) No. of people found');
xlabel(ax2,'iteration'); 
yyaxis(ax2,'left');  ylabel(ax2,'Lloyd Coverage Cost (J)');
yyaxis(ax2,'right'); ylabel(ax2,'No. of people found');

% ---- Setup animation recording (match Model 2 style) ----
video_filename = 'model3_seed30.avi';
try
    writerObj = VideoWriter(video_filename, 'Motion JPEG AVI');
    writerObj.FrameRate = 1/dt;   % ~8.33 fps for dt=0.12
    writerObj.Quality   = 90;
    open(writerObj);
    fprintf('Recording animation to: %s\n', video_filename);
catch ME
    fprintf('Warning: Could not create video file (%s). Animation will only display.\n', ME.message);
    writerObj = [];
end

% per-robot phase for sweep drift
robot_phi = linspace(0,2*pi,N)';

% ---- flags for "all found" and stopping ----
all_found       = false;
all_found_iter  = NaN;       % iteration when last target was found
final_iteration = T;         % iteration when simulation actually stopped
stop_iter       = T;         % iteration to stop at (all_found_iter + extra_steps)
t_last          = 0;         % last completed iteration index

%% ---------------- Main loop ----------------
for t = 1:T
    t_last = t;

    % ---- if we've already found all and finished the extra window, stop ----
    if all_found && t > stop_iter
        final_iteration = stop_iter;
        break;
    end
    
    % ---- bounded Voronoi via ghosts ----
    margin = 5;
    ghosts = [box(1,1)-margin, box(2,1)-margin;
              box(1,2)+margin, box(2,1)-margin;
              box(1,2)+margin, box(2,2)+margin;
              box(1,1)-margin, box(2,2)+margin];
    P_ext = sanitize_points([P; ghosts], box);
    [V, Cfull] = voronoin(P_ext); 
    C = Cfull(1:N);

    % ---- smooth rho & gradient ----
    rho_s = conv2(rho, K_smooth, 'same');
    [Gy, Gx] = gradient(rho_s);

    % ---- robust centroids & workloads ----
    Ccent = P; 
    s     = zeros(N,1);
    for i = 1:N
        idx = C{i}(C{i} > 1);
        if numel(idx) < 3, continue; end
        verts = V(idx,:);
        try
            Kh   = convhull(verts(:,1), verts(:,2));
            poly = intersect(polyshape(verts(Kh,1), verts(Kh,2)), Bpoly);
            if isempty(poly.Vertices), continue; end
            v = poly.Vertices;
            in = inpolygon(xg, yg, v(:,1), v(:,2));
            w  = rho(in); 
            if isempty(w), continue; end
            m = sum(w,'all')*dA + 1e-12;
            x = xg(in); 
            y = yg(in);
            Ccent(i,:) = [sum(w.*x,'all')*dA/m, sum(w.*y,'all')*dA/m];
            s(i)       = m;
        catch
        end
    end
    s_mean = mean(s(isfinite(s))); 
    if ~isfinite(s_mean), s_mean = 0; end

    % ---- frontier weight ----
    w_front = 1 ./ (1 + exp(-(s - s_mean)/max(tau_s,1e-6)));

    % ---- frontier direction ----
    F_frontier = zeros(N,2);
    for i = 1:N
        idx = C{i}(C{i}>1); 
        if numel(idx) < 3, continue; end
        verts = V(idx,:);
        try
            Kh = convhull(verts(:,1), verts(:,2));
            v  = verts(Kh,:); 
            in = inpolygon(xg, yg, v(:,1), v(:,2));
            g  = [mean(Gx(in),'all','omitnan'), ...
                  mean(Gy(in),'all','omitnan')];
            F_frontier(i,:) = g / (norm(g)+1e-9);
        catch
        end
    end

    % ---- inter-robot smooth repulsion ----
    R = zeros(N,2);
    for i = 1:N
        di = P(i,:) - P; 
        ri = sqrt(sum(di.^2,2)) + 1e-9;
        wi = exp(-ri/d0); 
        wi(i) = 0;
        R(i,:) = (wi.' * (di ./ ri));
    end

    % ---- WORKLOAD CONSENSUS (ENABLED) ----
    F_cons = zeros(N,2);
    for i = 1:N
        dP = P - P(i,:); 
        r2 = sum(dP.^2,2) + 1e-9;
        delta = tanh((s - s(i))./max(tau_c,1e-6)); 
        delta(i) = 0;
        F_cons(i,:) = sum((delta./r2).*dP, 1);
    end

    % ---- smooth wall (tanh inward) ----
    xmin=box(1,1); xmax=box(1,2); 
    ymin=box(2,1); ymax=box(2,2);
    X = P(:,1); Y = P(:,2);
    Wx = k_wall*( tanh((X - xmin)/bw_wall) - tanh((xmax - X)/bw_wall) );
    Wy = k_wall*( tanh((Y - ymin)/bw_wall) - tanh((ymax - Y)/bw_wall) );
    W  = [Wx, Wy];

    % ---- NO found person avoidance ----

    % ================== CONTROL LAW ==================
    U_det = k_lloyd*(Ccent - P) ...
          + k_frontier*(w_front .* F_frontier) ...
          + k_consensus*F_cons ...  % WITH workload sharing
          + k_repulse*R ...
          - W ...
          - k_center*(P - center);

    % adaptive anti-stall dither
    magU = sqrt(sum(U_det.^2,2));
    g_d  = 1 ./ (1 + exp((magU - u0)/gamma));
    noise = (eps_dither * [g_d g_d]) .* randn(N,2);

    % tiny rotating sweep drift
    D_sweep = zeros(N,2);
    D_sweep(:,1) = k_sweep * cos(omega*t + robot_phi);
    D_sweep(:,2) = k_sweep * sin(omega*t + robot_phi);

    U = U_det + noise + D_sweep;
    % ================================================

    % ---- integrate & sanitize ----
    P = P + dt*U;
    P = sanitize_points(P, box);

    % ---- update rho: decay near robots ONLY ----
    for i = 1:N
        d2 = (xg - P(i,1)).^2 + (yg - P(i,2)).^2;
        rho = rho .* exp(-0.025 * exp(-d2/2));
    end
    rho = 0.96*rho + 0.04*conv2(rho, K_smooth, 'same');

    % ---- detection ----
    for m = 1:M
        if ~found(m)
            d = vecnorm(P - Targets(m,:), 2, 2);
            if any(d <= sense_R)
                found(m) = true; 
                found_time(m) = t*dt;
                FoundCoords(end+1,:) = Targets(m,:); %#ok<AGROW>
                fprintf('FOUND target %d at (%.2f, %.2f) at t=%.2fs\n', ...
                        m, Targets(m,1), Targets(m,2), found_time(m));
            end
        end
    end
    num_found_hist(t) = sum(found);

    % ---- if all targets just got found, schedule stop_iter ----
    if ~all_found && num_found_hist(t) == M
        all_found      = true;
        all_found_iter = t;
        stop_iter      = min(T, all_found_iter + extra_steps_after_all_found);
        fprintf('\nALL TARGETS FOUND at iteration %d (t = %.1f s).\n', ...
                all_found_iter, all_found_iter*dt);
        fprintf('Continuing simulation for +%.1f s (until iteration %d).\n', ...
                extra_time_after_all_found, stop_iter);
    end

    % ---- coverage cost ----
    Jhist(t) = coverage_cost_lloyd(C, V, Bpoly, xg, yg, rho, dA);

    % ---- plotting ----
    if ~isvalid(ax1) || ~isvalid(ax2)
        f = figure('Color','w','Name',figName, 'Position', [100 100 1400 600]);
        tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
        ax1 = nexttile(1); axis(ax1,'equal'); grid(ax1,'on');
        xlim(ax1, box(1,:)); ylim(ax1, box(2,:));
        xlabel(ax1,'x'); ylabel(ax1,'y'); title(ax1,'Coverage, trajectories, targets');
        ax2 = nexttile(2); hold(ax2,'on'); grid(ax2,'on');
        title(ax2,'(Left) J(t)  |  (Right) No. of people found');
        xlabel(ax2,'iteration'); 
        yyaxis(ax2,'left');  ylabel(ax2,'Lloyd Coverage Cost (J)');
        yyaxis(ax2,'right'); ylabel(ax2,'No. of people found');
    end

    cla(ax1); hold(ax1,'on');
    imagesc(ax1, [min(xg(:)) max(xg(:))], [min(yg(:)) max(yg(:))], rho);
    set(ax1,'YDir','normal'); 
    colormap(ax1,'parula'); 
    alpha(ax1,0.35);

    % Voronoi cells (clipped)
    for i = 1:N
        idx = C{i}(C{i}>1); 
        if numel(idx) < 3, continue; end
        verts = V(idx,:);
        try
            Kh   = convhull(verts(:,1), verts(:,2));
            poly = intersect(polyshape(verts(Kh,1), verts(Kh,2)), Bpoly);
            if area(poly) > 1e-6
                plot(ax1, poly, 'FaceColor',[0.6 0.9 1.0], ...
                    'FaceAlpha',0.18,'EdgeColor',[0 0.35 0.8],'LineWidth',1.0);
            end
        catch
        end
    end

    % Trails & robots
    Ptrail(t,:,:) = P;
    for i = 1:N
        xy = reshape(Ptrail(1:t,i,:), [t,2]);
        plot(ax1, xy(:,1), xy(:,2), '-', 'Color', trail_colors(i,:), 'LineWidth',1.25);
    end
    plot(ax1, P(:,1), P(:,2), 's', 'MarkerFaceColor','y', ...
         'MarkerEdgeColor','k', 'LineWidth',1.0);

    % Targets (NO exclusion disks since no location sharing)
    for m = 1:M
        if found(m), fc=[0.00 0.70 0.00]; ec=[0.00 0.40 0.00];
        else,        fc=[0.85 0.00 0.00]; ec=[0.50 0.00 0.00];
        end
        draw_circle(ax1, Targets(m,:), 0.25, fc, ec, 1.2);
    end
    title(ax1, sprintf('Workload Balancing Only (No Location Sharing)\n t = %.1f s, found %d/%d', ...
           t*dt, sum(found), M));

    % Right plot with full history (like Model 2)
    cla(ax2); hold(ax2,'on'); grid(ax2,'on');
    yyaxis(ax2,'left'); 
    plot(ax2, 1:t, Jhist(1:t), 'b-', 'LineWidth',1.25); 
    ylabel(ax2,'J');
    yyaxis(ax2,'right'); 
    plot(ax2, 1:t, num_found_hist(1:t), 'r--', 'LineWidth',1.25); 
    ylabel(ax2,'found');
    xlabel(ax2,'iteration');
    
    drawnow limitrate;

    % Capture frame for animation
    if ~isempty(writerObj)
        try
            frame = getframe(f);
            writeVideo(writerObj, frame);
        catch ME
            warning('Error writing video frame at t=%d: %s', t, ME.message);
        end
    end
end

% If we never broke out of the loop, set final_iteration to last t
if final_iteration == T && t_last > 0
    final_iteration = t_last;
end

%% ---- Final frame handling & plots ----
t_final = final_iteration;

cla(ax1); hold(ax1,'on');
imagesc(ax1, [min(xg(:)) max(xg(:))], [min(yg(:)) max(yg(:))], rho);
set(ax1,'YDir','normal'); 
colormap(ax1,'parula'); 
alpha(ax1,0.35);

% Voronoi cells (use last C/V for visualization)
for i = 1:N
    idx = C{i}(C{i}>1); 
    if numel(idx) < 3, continue; end
    verts = V(idx,:);
    try
        Kh   = convhull(verts(:,1), verts(:,2));
        poly = intersect(polyshape(verts(Kh,1), verts(Kh,2)), Bpoly);
        if area(poly) > 1e-6
            plot(ax1, poly, 'FaceColor',[0.6 0.9 1.0], ...
                'FaceAlpha',0.18,'EdgeColor',[0 0.35 0.8],'LineWidth',1.0);
        end
    catch
    end
end

for i = 1:N
    xy = reshape(Ptrail(1:t_final,i,:), [t_final,2]);
    plot(ax1, xy(:,1), xy(:,2), '-', 'Color', trail_colors(i,:), 'LineWidth',1.25);
end
plot(ax1, P(:,1), P(:,2), 's', 'MarkerFaceColor','y', ...
     'MarkerEdgeColor','k', 'LineWidth',1.0);

for m = 1:M
    if found(m), fc=[0.00 0.70 0.00]; ec=[0.00 0.40 0.00];
    else,        fc=[0.85 0.00 0.00]; ec=[0.50 0.00 0.00];
    end
    draw_circle(ax1, Targets(m,:), 0.25, fc, ec, 1.2);
end
title(ax1, sprintf('Workload Balancing Only (No Location Sharing)\n t = %.1f s, found %d/%d', ...
       t_final*dt, sum(found), M));

cla(ax2); hold(ax2,'on'); grid(ax2,'on');
yyaxis(ax2,'left'); 
plot(ax2, 1:t_final, Jhist(1:t_final), 'b-', 'LineWidth',1.25); 
ylabel(ax2,'J');
yyaxis(ax2,'right'); 
plot(ax2, 1:t_final, num_found_hist(1:t_final), 'r--', 'LineWidth',1.25); 
ylabel(ax2,'found');
xlabel(ax2,'iteration');

drawnow;

% Add some still frames at the end for nicer video ending
if ~isempty(writerObj)
    try
        for i = 1:round(writerObj.FrameRate)  % ~1 second of still
            frame = getframe(f);
            writeVideo(writerObj, frame);
        end
    catch ME
        warning('Error writing final still frames: %s', ME.message);
    end
end

% Close video writer
if ~isempty(writerObj)
    close(writerObj);
    fprintf('Animation saved to: %s\n', video_filename);
end

% print final results
fprintf('\n=== MODEL 3 RESULTS ===\n');

if all_found
    fprintf('All targets found at: %.1f seconds (iteration %d)\n', ...
            all_found_iter*dt, all_found_iter);
else
    fprintf('All targets NOT found within horizon.\n');
end

fprintf('Simulation ended at: %.1f seconds (iteration %d)\n', ...
        t_final*dt, t_final);
fprintf('Targets found: %d/%d\n', sum(found), M);

if ~isempty(FoundCoords)
    disp(array2table(FoundCoords, 'VariableNames',{'x','y'}));
end
disp(table((1:M)', found, found_time, ...
     'VariableNames',{'person','found','time_sec'}));

%% ---------------- Helpers ----------------
function P = sanitize_points(P, box)
    bad = any(~isfinite(P),2);
    if any(bad)
        nx = sum(bad);
        P(bad,1) = box(1,1) + (box(1,2)-box(1,1))*rand(nx,1);
        P(bad,2) = box(2,1) + (box(2,2)-box(2,1))*rand(nx,1);
    end
    epsb = 1e-6;
    P(:,1) = max(min(P(:,1), box(1,2)-epsb), box(1,1)+epsb);
    P(:,2) = max(min(P(:,2), box(2,2)-epsb), box(2,1)+epsb);
end

function J = coverage_cost_lloyd(C, V, Bpoly, X, Y, W, dA)
    J = 0;
    for i = 1:numel(C)
        idx = C{i}(C{i}>1); 
        if numel(idx) < 3, continue; end
        verts = V(idx,:);
        try
            Kh   = convhull(verts(:,1), verts(:,2));
            poly = intersect(polyshape(verts(Kh,1), verts(Kh,2)), Bpoly);
            if isempty(poly.Vertices) || area(poly) < 1e-10, continue; end
            v  = poly.Vertices; 
            in = inpolygon(X, Y, v(:,1), v(:,2));
            w  = W(in); 
            if isempty(w), continue; end
            m  = sum(w,'all')*dA + 1e-12;
            x  = X(in); 
            y  = Y(in);
            cx = sum(w.*x,'all')*dA/m; 
            cy = sum(w.*y,'all')*dA/m;
            dx = X(in) - cx; 
            dy = Y(in) - cy;
            J  = J + sum((dx.^2 + dy.^2).*w,'all') * dA;
        catch
        end
    end
end

function draw_circle(ax, c, r, fc, ec, lw)
    th = linspace(0,2*pi,60);
    plot(ax, c(1)+r*cos(th), c(2)+r*sin(th), '-', ...
         'Color', ec, 'LineWidth', lw);
    patch('XData', c(1)+r*cos(th), 'YData', c(2)+r*sin(th), ...
          'FaceColor', fc, 'EdgeColor','none', ...
          'FaceAlpha',0.9, 'Parent', ax);
end

end