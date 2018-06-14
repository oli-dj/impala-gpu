folderstring = 'june13_compare_trim%i_randpath';

trims = [0 1 2 5];

figure;

for i = 1:length(trims)
    
    %Load
    load([sprintf(folderstring,trims(i)) '//setup.mat']);
    s1 = subplot(1,4,1);
    plot(1:900,stats.template_length)
    hold on;
    s2 = subplot(1,4,2);
    plot(1:900,stats.informed_final,'.')
    alpha(.5)
    hold on;
    s3 = subplot(1,4,3);
    plot(1:900,stats.informed_init-stats.informed_final,'.')
    hold on;
    
    s4 = subplot(1,4,4);
    plot(1:900,stats.time_elapsed)
    hold on;
end

%%
title(s1, 'Template size');
title(s2, 'Final Conditional data');
title(s3, 'Library Mismatches');
title(s4, 'Commulative iteration time');

grid(s1,'on');
grid(s2,'on');
grid(s3,'on');

xlabel(s1,'Iteration');
xlabel(s2,'Iteration');
xlabel(s3,'Iteration');
xlabel(s4,'Iteration');

%ylabel(s1,'');
%set(s2,'ytick',[]);
%set(s3,'ytick',[]);
set(s2,'yticklabel',[]);
set(s3,'yticklabel',[]);
set(s4,'YaxisLocation','right');
ylabel(s4,'Time [$s$]');

axis(s1,[0 900 0 260]);
axis(s2,[0 900 0 260]);
axis(s3,[0 900 0 260]);
axis(s4,[0 900 0 27]);

legend(s4,'0','1','2','5');
legend('Location','NorthWest');