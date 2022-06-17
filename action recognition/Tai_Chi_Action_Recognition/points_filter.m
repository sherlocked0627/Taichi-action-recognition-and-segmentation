function traj = points_filter(A,num)

traj = A;
a = size(A,2)
while num > 0
    for t = 5:a-4;
        for k = 1:3
            for i = 1:4
                traj(k,t) = traj(k,t) + A(k,t+i) + A(k,t-i);
            end
            traj(k,t) = traj(k,t)/9;
        end
    end
    num = num-1;
end
