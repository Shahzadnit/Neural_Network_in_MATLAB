clc;
clear all;
load fisheriris;

%%%%%%%%%%%%%%%%%%%%DATASET%%%%%%%%%%%%%%

meas;
dataset=meas';
for j=1:4
    dataset(j,:)=(dataset(j,:)-mean(dataset(j,:)))/std(dataset(j,:));
end

trainset=zeros(5,105);
testset=zeros(5,45);
for i=1:35
    trainset(5,i)=1;
    trainset(5,i+35)=2;
    trainset(5,i+70)=3;
end
for i=1:15
    testset(5,i)=1;
    testset(5,i+15)=2;
    testset(5,i+30)=3;
end


r=randperm(50);
for i=1:35
    trainset(1:4,i)=dataset(:,r(i));
    trainset(1:4,i+35)=dataset(:,r(i)+50);
    trainset(1:4,i+70)=dataset(:,r(i)+100);
    end
for i=1:15
   testset(1:4,i)=dataset(:,r(i+35));
   testset(1:4,i+15)=dataset(:,r(i+35)+50);
   testset(1:4,i+30)=dataset(:,r(i+35)+100);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


batch_size=25;
in_neuron=4;
neuron_at_1=10;
neuron_at_2=8;
neuron_at_output=3;
epoch=500;
 
%w1=rand(neuron_at_1,in_neuron);
w1=random('normal',0,1,neuron_at_1,in_neuron);
b1=rand(neuron_at_1,1);
%b1=random('normal',0,1,neuron_at_1,1);
%w2=rand(neuron_at_2,neuron_at_1);
w2=random('normal',0,1,neuron_at_2,neuron_at_1);
b2=rand(neuron_at_2,1);
%b2=random('normal',0,1,neuron_at_2,1);
%w3=rand(neuron_at_output,neuron_at_2);
w3=random('normal',0,1,neuron_at_output,neuron_at_2);
b3=rand(neuron_at_output,1);
%b3=random('normal',0,1,neuron_at_output,1);
a1=zeros(neuron_at_1,batch_size);
a2=zeros(neuron_at_2,batch_size);
a3=zeros(neuron_at_output,batch_size);
a11=zeros(neuron_at_1,1);
a22=zeros(neuron_at_2,1);
a33=zeros(neuron_at_output,1);
t1=[1;0;0];
t2=[0;1;0];
t3=[0;0;1];
e1=[0;0;0];
tar1=zeros(3,105);
tar2=zeros(3,45);
for i=1:35
    tar1(:,i)=[1;0;0];
    tar1(:,i+35)=[0;1;0];
    tar1(:,i+70)=[0;0;1];
end
for i=1:15
    tar2(:,i)=[1;0;0];
    tar2(:,i+15)=[0;1;0];
    tar2(:,i+30)=[0;0;1];
end

 e=zeros(neuron_at_output,batch_size);

 eta=0.01;
 alpa=0.001;

 


 x=zeros(1,epoch);
 y=zeros(1,epoch);
 y2=zeros(1,epoch);

 acc1=0;
 %flag=0;

no_of_batch=105/batch_size;
tic;
prev_acc=0;
cre1=0;

cre_v=0;
pl=1;
for j1=1:epoch
    d=randperm(105);
    cre1=0;
for i=1:no_of_batch
%%%%%%%%%%%%%%%Forward-propagation%%%%%%%%%%%%%%%%%%%%%%%%%%
    a1=logsig(w1*trainset([1 2 3 4 ],d(i:batch_size+i-1))+repmat(b1,1,batch_size));
    a2=logsig(w2*a1+repmat(b2,1,batch_size));
    a3=softmax(w3*a2+repmat(b3,1,batch_size));
         
    e=a3-tar1(:,d(i:batch_size+i-1));
    err=mean(mean(e.*e));
    cre1=cre1+err;
%%%%%%%%%%%%%%%Back-propagation%%%%%%%%%%%%%%%%%%%%%%%%%% 

    delta3=e.*(a3.*(1-a3));
    dw3=delta3*a2';
    
               
    delta2=(w3'*delta3).*(a2.*(1-a2));
    dw2=delta2*a1';
    
               
    delta1=(w2'*delta2).*(a1.*(1-a1));
    dw1=delta1*trainset([1 2 3 4 ],d(i:batch_size+i-1))';
    
    %%%%%%%%UPDATE_WEIGHT and  Regularization %%%%%%%%%%%%%%%%
%     w3 = w3-(eta.*dw3);
%     w2=w2-(eta.*dw2);
%     w1=w1-(eta.*dw1);
%       
%     b3 = b3-(eta.*sum(delta3,2));
%     b2=b2-(eta.*sum(delta2,2));
%     b1=b1-(eta.*sum(delta1,2));

    w3 = w3-(eta.*dw3)-(alpa*eta).*w3;
    w2 = w2-(eta.*dw2)-(alpa*eta).*w2;
    w1 = w1-(eta.*dw1)-(alpa*eta).*w1;

    b3 = b3-(eta.*sum(delta3,2))-(alpa*eta).*b3;
    b2=b2-(eta.*sum(delta2,2))-(alpa*eta).*b2;
    b1=b1-(eta.*sum(delta1,2))-(alpa*eta).*b1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
end
%%%%%%%%%%%%%%%%%%%Validation Testing%%%%%%%%%%%%%%%%%%%%%%
count=0;
for k=1:45
    cre2=0;
    a11 = logsig( w1*testset(1:4,k)+b1 );
    a22 = logsig( w2*a11 +b2);
    a33 = softmax( w3*a22 +b3 );
    class=find(a33 == max(a33(:)));
    if (k<16 && class==1)
        e1=a33-t1;
        cre_v=(e1'*e1)/2;
        
        count=count+1;
    end
    if (15<k&&k<31&& class==2)
        e1=a33-t2;
        cre_v=(e1'*e1)/2;
        
        count=count+1;
    end
    if (30<k&&k<46 && class==3)
        e1=a33-t3;
        cre_v=(e1'*e1)/2;
        count=count+1;
    end
end
acc=0;
acc=(count/45);
acc1=acc1+acc;
y3(1,j1)=cre_v;


y2(1,j1)=acc1;
acc1=0;
x(1,j1)=j1;
y1(1,j1)=cre1/no_of_batch;

cre_v=0;

plot(x(1:j1),y1(1:j1),'r',x(1:j1),y2(1:j1),'g',x(1:j1),y3(1:j1),'b')
ylim([-0.01 1.1]);
xlabel('no of epoch')
ylabel('Accuracy,Training error and Validation error')
legend('Training error','Test Accuracy','Validation error')
grid
drawnow;

 end
time1 = toc


count=0;
for k=1:45
    a11 = logsig( w1*testset(1:4,k)+b1 );
    a22 = logsig( w2*a11 +b2);
    a33 = softmax( w3*a22 +b3 );
    class=find(a33 == max(a33(:)));
    if (k<16 && class==1)
        
        count=count+1;
    end
    if (15<k&&k<31&& class==2)
        
        count=count+1;
    end
    if (30<k&&k<46 && class==3)
        
        count=count+1;
    end
end

acc=(count/45)*100

