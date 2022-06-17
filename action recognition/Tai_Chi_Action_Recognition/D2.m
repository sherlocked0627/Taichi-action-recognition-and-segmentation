function result = D2(a,b,c,d)

cd = [d(1,1)-c(1,1);d(2,1)-c(2,1);d(3,1)-c(3,1)];
ab = [b(1,1)-a(1,1);b(2,1)-a(2,1);b(3,1)-a(3,1)];

F = [ab,cd];

if rank(F) < 2
    result = 0;
else
    cos_abcd = dot(ab,cd) / (norm(ab)*norm(cd));
    if cos_abcd > 0
        sin_abcd = sqrt(1 - cos_abcd.^2);
        result = norm(cd) * sin_abcd;
    else
        result = norm(cd);
    end
end
    
