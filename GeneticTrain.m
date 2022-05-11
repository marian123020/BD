function [bestfis results]=GeneticTrain(fis,data)

    %% Problemos apibrėžimas
    p0=FISParameters(fis);  
    Problem.CostFunction=@(x) FISCost(x,fis,data);    
    Problem.nVar=numel(p0);    
    alpha=1;
    Problem.VarMin=-(10^alpha);
    Problem.VarMax=10^alpha;

    % GA Parametrai
    Params.nPop=7;

    % Leidžiamas GA 
    results=RunGA(Problem,Params);
    
    % Gaunamas rezultatas
    p=results.BestSol.Position.*p0;
    bestfis=FISSet(fis,p);
    
end

function results=RunGA(Problem,Params)

    disp('Genetic Algorithm is Started');

    % Problemos apibrėžimas
    CostFunction=Problem.CostFunction;        % Kainos funkcija
    nVar=Problem.nVar;          % Sprendimo kintamųjų skaičius
    VarSize=[1 nVar];           % Sprendimo kintamųjų matricos dydis
    VarMin=Problem.VarMin;      % Apatinė kintamųjų riba
    VarMax=Problem.VarMax;      % Viršutinė kintamųjų riba

    % GA Parametrai

    nPop=Params.nPop;        % Populiacijos dydis
    pc=0.6;                 % Kryžminis procentas
    nc=2*round(pc*nPop/2);  % Palikuonių skaičius (tėvai)
    pm=0.5;                 % Mutacijos procentas
    nm=round(pm*nPop);      % Mutantijos skaičius
    gamma=0.2;
    mu=0.1;         % Mutacijos greitis
    beta=8;         % Pasirinkimo slėgis

    % Inicijavimas

    empty_individual.Position=[];
    empty_individual.Cost=[];

    pop=repmat(empty_individual,nPop,1);

    for i=1:nPop

        % Inicijuoti poziciją
        if i>1
            pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
        else
            pop(i).Position=ones(VarSize);
        end
        % Įvertinimas
        pop(i).Cost=CostFunction(pop(i).Position);

    end

    % Rūšiuoti populiaciją
    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder);

    % Geriausias sprendimas
    BestSol=pop(1);

    % Masyvas geriausioms sąnaudų vertėms
    BestCost=zeros(1,1);

    % Blogiausia kaina
    WorstCost=pop(end).Cost;
    % Pagrindinė funkcija
	
	check = false;

    while check == false
        P=exp(-beta*Costs/WorstCost);
        P=P/sum(P);

        % Kryziavimas
        popc=repmat(empty_individual,nc/2,2);
        for k=1:nc/2
            % Pasirinkimas tevų indeksas
            i1=RouletteWS(P);
            i2=RouletteWS(P);
            % Pasirenkami tevai
            p1=pop(i1);
            p2=pop(i2);
            % Taisomas kryžiavimas
            [popc(k,1).Position, popc(k,2).Position]=...
                Crossover(p1.Position,p2.Position,gamma,VarMin,VarMax);
            % Įvertinama palikuonius
            popc(k,1).Cost=CostFunction(popc(k,1).Position);
            popc(k,2).Cost=CostFunction(popc(k,2).Position);

        end
        popc=popc(:);
        % Mutacija
        popm=repmat(empty_individual,nm,1);
        for k=1:nm

            % Pasirenkami tevai
            i=randi([1 nPop]);
            p=pop(i);
            % Mutacijos taisymas
            popm(k).Position=Mutate(p.Position,mu,VarMin,VarMax);
            % Mutacijos palyginimas
            popm(k).Cost=CostFunction(popm(k).Position);
        end

        % Sukurti sujungtą populiaciją
        pop=[pop
             popc
             popm]; 

        % Rūšiuoti populiaciją
        Costs=[pop.Cost];
        [Costs, SortOrder]=sort(Costs);
        pop=pop(SortOrder);

        % Atnaujinkite blogiausią kainą
        WorstCost=max(WorstCost,pop(end).Cost);

        % Sutrumpinimas
        pop=pop(1:nPop);
        Costs=Costs(1:nPop);
        % Masyve geriausias kada nors rastas sprendimas
        BestSol=pop(1);
        % Masyve kaina geriausia kada nors atrasta
        BestCost(length(BestCost)+1)=BestSol.Cost;
        % Rodyti iteracijos informaciją
        disp(['In Iteration Number ' num2str(length(BestCost))-1 ': Highest Cost Is = ' num2str(BestCost(end))]);
		check = checkBestSol(BestCost(end-1),BestCost(end));
    end
	BestCost = BestCost(2:end);
	
    disp('Genetic Algorithm is Finished');
    
    %% Rezultatas

    results.BestSol=BestSol;
    results.BestCost=BestCost;
	
    
end

function [y1, y2]=Crossover(x1,x2,gamma,VarMin,VarMax)

    alpha=unifrnd(-gamma,1+gamma,size(x1));
    
    y1=alpha.*x1+(1-alpha).*x2;
    y2=alpha.*x2+(1-alpha).*x1;
    
    y1=max(y1,VarMin);
    y1=min(y1,VarMax);
    
    y2=max(y2,VarMin);
    y2=min(y2,VarMax);

end

function y=Mutate(x,mu,VarMin,VarMax)

    nVar=numel(x);
    
    nmu=ceil(mu*nVar);
    
    j=randsample(nVar,nmu)';
    
    sigma=0.1*(VarMax-VarMin);
    
    y=x;
    y(j)=x(j)+sigma*randn(size(j));
    
    y=max(y,VarMin);
    y=min(y,VarMax);

end