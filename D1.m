function result = D1(a,b,c,d)

ac = [c(1,1)-a(1,1);c(2,1)-a(2,1);c(3,1)-a(3,1)];
ad = [d(1,1)-a(1,1);d(2,1)-a(2,1);d(3,1)-a(3,1)];
ab = [b(1,1)-a(1,1);b(2,1)-a(2,1);b(3,1)-a(3,1)];

if ac == 0
    l1 = 0;
    cos_abd = dot(ad,ab) / (norm(ad)*norm(ab));
    L2 = abs(ad * cos_abd);
    l2 = sqrt(L2(1,1).^2 + L2(2,1).^2 + L2(3,1).^2);
    result = (l1.^2 + l2.^2) / (l1 + l2);
else
    cos_abc = dot(ac,ab) / (norm(ac)*norm(ab));
    L1 = abs(ac * cos_abc);
    l1 = sqrt(L1(1,1).^2 + L1(2,1).^2 + L1(3,1).^2);
    cos_abd = dot(ad,ab) / (norm(ad)*norm(ab));
    L2 = abs(ad * cos_abd);
    l2 = sqrt(L2(1,1).^2 + L2(2,1).^2 + L2(3,1).^2);
    result = (l1.^2 + l2.^2) / (l1 + l2);
end
   
