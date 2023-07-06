clear all;
close all;
clc;

Num_Beams=129; % Number of beams transmitted for each PA frame
R_Init=0.001; % Minimal radius [m]
R_Res=0.00015; % Radial resolution [m]
Num_R=380;%462; % Number of radial grid elements
Sin_Theta_X_Init=-0.699999988; % Minimal sin(angle)
Sin_Theta_X_Res=0.014; % Angular resolution (actually resolution of Sin(Theta_X) )
R=R_Init+(0:R_Res:(R_Res*(Num_R-1))); % Radial axis
Sin_Theta_X=Sin_Theta_X_Init+(0:Sin_Theta_X_Res:Sin_Theta_X_Res*(Num_Beams-2)); % Polar axis (again actually Sin(Theta_X) )

[R_Mat, Sin_Theta_X_Mat] = meshgrid(R, Sin_Theta_X); % Make grid
R_Mat=R_Mat'; Sin_Theta_X_Mat=Sin_Theta_X_Mat'; % Make matrices be in the currect orientation
[x,y] = pol2cart(asin(Sin_Theta_X_Mat),R_Mat); % Convert to Cartesian coordinates   

% save("./plot3d/plot3d-sim.mat", "x", "y");