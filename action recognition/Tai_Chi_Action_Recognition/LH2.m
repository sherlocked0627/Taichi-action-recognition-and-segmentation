function result = LH2(start,curr,P)

plus = 0;

if curr == 2
    result = log2(norm(P(:,2) - P(:,1)));
else
    for j = start : curr-1
         plus = plus + norm(P(:,j+1)-P(:,j));
    end
    result = log2(plus);
end

