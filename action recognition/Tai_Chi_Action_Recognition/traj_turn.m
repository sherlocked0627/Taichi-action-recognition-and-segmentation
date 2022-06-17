function E = traj_turn(A1,traj,leftshoulder,rightshoulder,i)
A2 = traj{rightshoulder,i};
A3 = traj{leftshoulder,i};
m = 50;
B1 = A1;
B2 = points_filter(A2,m); 
B3 = points_filter(A3,m); 

a = size(A1,2);


aver = round((a-1)/2);
lra1 = [0 -1];
lrb1_1 = B2(1,aver) - B3(1,aver);
lrb1_2 = B2(3,aver) - B3(3,aver);
lrb1 = [lrb1_1 lrb1_2];
site1 = acos(dot(lra1,lrb1)/(norm(lra1)*norm(lrb1)))*180/pi;


E = zeros(3,a-9);
for r1 = 1:a-9
     E(1,r1) = (B1(1,r1+4) - B3(1,5))*cosd(site1) - (B1(3,r1+4) - B3(3,5))*sind(site1) + B3(1,5);
     E(2,r1) = B1(2,r1+4);
     E(3,r1) = (B1(1,r1+4) - B3(1,5))*sind(site1) + (B1(3,r1+4) - B3(3,5))*cosd(site1) + B3(3,5);
end
