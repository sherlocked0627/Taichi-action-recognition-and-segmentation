function result = nozeronumber(C)

a = length(C);

result = 0;

for i = 1 : a
    if C(1,i) ~= 0 && C(2,i) ~= 0 && C(3,i) ~= 0
        result = result + 1;
    end
end
    