% actor critic model
gamma=.95; %discount rate
alpha=.25; %learning rate

%diff from td learning since it's not single feedforward network where
%you're computing value function or output, one network creates values and
%another predicts action. so define 2 networks
% use 10x10 grid world example. this constitutes 100 different states for
% agent
grid_D=10; %dimensions of grid world
reward_state=100; %where is reward in the world, 100= 10x10 area
beginning_state=1; %where agent starts in world 1= 1x1 area

moveN=4; % number of possible actions
actor_network=rand(grid_D.^2,moveN).*0.001; % states x actions produce an action (up, down, lt, rt) in grid world
critic_network=zeros(grid_D.^2, 1); % produce a single output prediction from critic layer

for i=1:1000 %number of trials, learning experiences in grid world
    % create world, reset state of world to pristine state
    world_state=zeros(grid_D); %create empty grid world
    world_state(beginning_state)=1; %put agent in to beginning of world
    not_rewarded=1;
    
    % may take a variable amount of time for agent to get to reward, so
    % diff from td learning where we set time steps static for each trial
    while not_rewarded==1
        % reshape grid world into a vector, to deliver to networks
        input=reshape(world_state,1,grid_D.^2);
        
        %get current coordinates of agent
        [x,y]=find(world_state==1); 
        
        %get action with some random noise included, multiply input by some
        %weight matrix to get some output
        action=input*actor_network + rand(1,4).*5; %random addition to ensure agent explores entire area
        prediction=input*critic_network;
        
%         %could put action through a softmax function instead:
%         action=input*actor_network;
%         beta=.1; %the gain, increasing it will enhance the highest action value, low action values go to 0, decreasing it means higher temperature, more randomness, brings probability of selecting any action more equal
%         soft_action=exp(beta.*action)./sum(exp(beta.*action));
        
        %handle edge cases- can't leave the world
        if x==1 %upper edge
            action(1)=-10;
        end
        if x==grid_D %lower edge
            action(2)=-10;
        end
        if y==1 %left edge
            action(3)=-10;
        end
        if y==grid_D %right edge
            action(4)=-10;
        end
        
        %select action based on the current most active unit
        this_action = find(action==max(action)); %choose action most valuable
        
        %update the world and agent's relation to it
        switch(this_action)
            case 1 %move up
                newx=x-1;
                newy=y;
            case 2%move down
                newx=x+1; newy=y;
            case 3%move left
                newy=y-1;
                newx=x;
            case 4 %move right
                newy=y+1;
                newx=x;
        end
        new_state=zeros(grid_D);
        new_state(newx,newy)=1; new_inp=reshape(new_state,1,grid_D.^2);
        %generate new prediction at the new state for TD learning
        new_pred=new_inp*critic_network;
        
        %check if there's a reward
        if new_inp(100)==1 %reward
            reward=1;
            not_rewarded=0;
        else
            reward=0;
        end
        % can put a punishment in certain parts of the grid world to
        % structure which way the agent goes. ie: could put a wall up in
        % certain part of world
%         if newy==6 &&
        
        %get TD prediction error
        error = reward+gamma.*new_pred-prediction;
        
        %now use error to train actor network
        actor_error=zeros(1,moveN);
        actor_error(this_action)=error; %error applies only to the selected
        delta_actor_weights=input'*actor_error;
        actor_network=actor_network+alpha.*delta_actor_weights;
        
        %now update the critic
        delta_critic_weights=input'*error;
        critic_network=critic_network+alpha.*delta_critic_weights;
        world_state=new_state;
        if mod(i,100)==1
            ws=world_state; ws(reward_state)=2;
            imagesc(ws)
            drawnow;
            pause(.01);
        end
    end
end
      

%visualizations- show this corollary to the value function
imagesc(reshape(critic_network,grid_D,grid_D))
colorbar
    