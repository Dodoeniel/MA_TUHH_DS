% Erzeugen von Testdaten zum Füttern eines LSTM Netzwerkes

fs = 40000;
t = 0:1/fs:(1-1/fs);
numExamples = 4;
numValid = 2;
A = zeros(numExamples+numValid,length(t));
%normaler Sinus
B = zeros(numExamples+numValid,1);
for i = 1:6
    l = rand(1);
    A(i,:) = sin(0.01*l*t);
    if abs(l) < 0.5
        B(i) = 1;
    end
end

% write on csv file
csvwrite('testData.csv',A)
csvwrite('labelData.csv',B)