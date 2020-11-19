close all;
clear;
clc;



neuron_min=10;
neuron_max=30;


%training pozymiu apskaiciavimas
pozymiai_tinklo_mokymui = pozymiai_raidems_atpazinti_begraph('training4.png', 8,false);




%test pozymiu apskaiciavimas
names={'0102','1745','4321','5678','7452','69','72','3333','1678'};
test_len=9;%test len
for i=1:test_len
    pavadinimas = strcat(names{i},'.png');
    %disp(pavadinimas);
    test_pozymiai{i}= pozymiai_raidems_atpazinti_begraph(pavadinimas, 1,0);
end

%net=apmokyti_ff(pozymiai_tinklo_mokymui,20);
%test_fun_ff(test_pozymiai{1},names{1},net,1);


%% radial basis network
for i=neuron_min:neuron_max
    clear tinkas;
    tinklas=apmokyti_newrb(pozymiai_tinklo_mokymui,i);
    
    result(i-neuron_min+1)=0;
    for ii=1:test_len
        result(i-neuron_min+1)=result(i-neuron_min+1)+test_fun_newrb(test_pozymiai{ii},names{ii},tinklas,0);
    end

end

%
disp('----------------');
for i=neuron_min:neuron_max
   fprintf("newrb neuronai %d  tikslumas  %.2f %%\n",i,result(i-neuron_min+1)/9*100)
end
figure();
plot(neuron_min:neuron_max,result,'-b');
hold on;

%% feedfowrward neural network
for i=neuron_min:neuron_max
    clear net;
    %tinklas=apmokyti(pozymiai_tinklo_mokymui,i);
    net=apmokyti_ff(pozymiai_tinklo_mokymui,i);
    
    result(i-neuron_min+1)=0;
    for ii=1:test_len
        result(i-neuron_min+1)=result(i-neuron_min+1)+test_fun_ff(test_pozymiai{ii},names{ii},net,0);
    end

end

%
disp('----------------');
for i=neuron_min:neuron_max
   fprintf("newff neuronai %d  tikslumas  %.2f %%\n",i,result(i-neuron_min+1)/9*100);
end
disp('----------------');
plot(neuron_min:neuron_max,result,'-r');
legend('newrb', ' newff');



disp('fin');
disp('is rezultatu matosi, kad arba as naudoju newff neteisingai arba jis reikalauka zymiai didesnio data set, kai newrb rezultatai yra stabilesni ir pakankamai tikslus. daznos klaidos atsiranda del mano negrazaus rasto))');
return

%% RB apmokymo funkcija
function tinklas=apmokyti_newrb(pozymiai_tinklo_mokymui,neuronai)
    
    P = cell2mat(pozymiai_tinklo_mokymui);
    T = [eye(10), eye(10), eye(10), eye(10), eye(10), eye(10), eye(10), eye(10)];
    tinklas = newrb(P,T,0,1,neuronai);
    %tinklas.trainParam.showWindow = 0;
    

end


%% RB testavimo funkcija
function out=test_fun_newrb(pozymiai_patikrai,test,tinklas,print)
    P2 = cell2mat(pozymiai_patikrai);
    Y2 = sim(tinklas, P2);
    [a2, b2] = max(Y2);
    raidziu_sk = size(P2,2);
    atsakymas = [];
    for k = 1:raidziu_sk
        atsakymas = [atsakymas, int2str(b2(k)-1)];
    end
    out=strcmp(atsakymas,test);
    if print
        if out
           result ='good';
        else
            result ='fail';
        end
        fprintf("newrb %s  result: %s  real: %s \n",result,atsakymas,test);
    end

end

%% FF apmokymo funkcija
function net=apmokyti_ff(pozymiai_tinklo_mokymui,neuronai)
    
    T = [eye(10), eye(10), eye(10), eye(10), eye(10), eye(10), eye(10), eye(10)];
    net = feedforwardnet([neuronai,ceil(neuronai/2)]);
    net.trainParam.showWindow = 0;
    P = cell2mat(pozymiai_tinklo_mokymui);
    net = train(net,P,T);


end


%% FF testavimo funkcija
function out=test_fun_ff(pozymiai_patikrai,test,net,print)
    P2 = cell2mat(pozymiai_patikrai);
    Y2 = net(P2);
    [a2, b2] = max(Y2);
    raidziu_sk = size(P2,2);
    atsakymas = [];
    for k = 1:raidziu_sk
        atsakymas = [atsakymas, int2str(b2(k)-1)];
    end
    out=strcmp(atsakymas,test);
    if print
        if out
           result ='good';
        else
            result ='fail';
        end
        fprintf("newff %s  result: %s  real: %s \n",result,atsakymas,test);
    end   
end