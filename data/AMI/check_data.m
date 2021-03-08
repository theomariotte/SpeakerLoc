clear; clc; %close all;
the_root = 'IS1000a/';
name = 'IS1000a';
%name = 'ES2002a';
for ii = 1 : 8
    
    [xx,fs] = audioread(sprintf('%s.Array1-0%d.wav',name,ii));
    N = length(xx);
    tt = (0:N-1)/fs;
    figure
    plot(tt,xx,'r');
    grid on
    
end