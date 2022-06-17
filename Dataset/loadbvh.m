function [skeleton,time] = loadbvh(fname)
%% LOADBVH  Load a .bvh (Biovision) file.
%����BVH�ļ������Բ����bvh���ļ���׺�������ļ�������ؽڵ�������Ϣ���Ҵ洢��SKELETON�ṹ����
% Some details on the BVH file structure are given in "Motion Capture File
% Formats Explained": http://www.dcs.shef.ac.uk/intranet/research/resmes/CS0111.pdf
% But most of it is fairly self-evident.

%% Load and parse header data
%
% The file is opened for reading, primarily to extract the header data (see
% next section). However, I don't know how to ask Matlab to read only up
% until the line "MOTION", so we're being a bit inefficient here and
% loading the entire file into memory. Oh well.

% add a file extension if necessary:
if ~strncmpi(fliplr(fname),'hvb.',4)  %fliplr���ҷ�ת����strncmpiƥ�䲻���ִ�Сд
  fname = [fname,'.bvh'];
end

fid = fopen(fname);
C = textscan(fid,'%s'); %��ʱ��CΪԪ������
fclose(fid);
C = C{1};   %C��ʱΪ�ַ���������


%% Parse data
%
% This is a cheap tokeniser, not particularly clever.
% Iterate word-by-word, counting braces and extracting data.

% Initialise:
skeleton = [];
ii = 1;     %��������C���������
nn = 0;     %�������Ĺؽڵ�����
brace_count = 1;

while ~strcmp( C{ii} , 'MOTION' )
  
  ii = ii+1;
  token = C{ii};
  
  if strcmp( token , '{' )
    
    brace_count = brace_count + 1;
    
  elseif strcmp( token , '}' )
    
    brace_count = brace_count - 1;
    
  elseif strcmp( token , 'OFFSET' )     %��¼ƫ�Ƶ�λ��
    
    skeleton(nn).offset = [str2double(C(ii+1)) ; str2double(C(ii+2)) ; str2double(C(ii+3))];
    ii = ii+3;
    
  elseif strcmp( token , 'CHANNELS' )    %��¼���ݵ�ͨ����
    
    skeleton(nn).Nchannels = str2double(C(ii+1));
    
    % The 'order' field is an index corresponding to the order of 'X' 'Y' 'Z'.
    % Subtract 87 because char numbers "X" == 88, "Y" == 89, "Z" == 90.
    % XYZ��Ӧ��ASCII��
    if skeleton(nn).Nchannels == 3
      skeleton(nn).order = [C{ii+2}(1),C{ii+3}(1),C{ii+4}(1)]-87;   %��ʾ������ת�Ĵ���
    elseif skeleton(nn).Nchannels == 6
      skeleton(nn).order = [C{ii+5}(1),C{ii+6}(1),C{ii+7}(1)]-87;
    else
      error('Not sure how to handle not (3 or 6) number of channels.')
    end
    
    if ~all(sort(skeleton(nn).order)==[1 2 3])
      error('Cannot read channels order correctly. Should be some permutation of [''X'' ''Y'' ''Z''].')
    end

    ii = ii + skeleton(nn).Nchannels + 1;

  elseif strcmp( token , 'JOINT' ) || strcmp( token , 'ROOT' )
    % Regular joint
    
    nn = nn+1;  %�ؽڵ�������
    
    skeleton(nn).name = C{ii+1};
    skeleton(nn).nestdepth = brace_count;

    if brace_count == 1    %һ���ڵ㣬���ڵ���߳�Ϊ���ڵ�
      % root node
      skeleton(nn).parent = 0;
    elseif skeleton(nn-1).nestdepth + 1 == brace_count; %���ӽڵ㣬ǰһ�ڵ����丸�ڵ㣨������������ӽڵ�����ڸ��ڵ�֮����֣�
      % if I am a child, the previous node is my parent:
      skeleton(nn).parent = nn-1;
    else
      % if not, what is the node corresponding to this brace count?
      % ���һ���ڵ㣬ǰһ���ڵ㲻�����ĸ��ڵ㣬�丸�ڵ����ϼ����ڵ�֮ǰ����ô���ݼ�¼������ֵ�жϸ��ڵ�
      prev_parent = skeleton(nn-1).parent;
      while skeleton(prev_parent).nestdepth+1 ~= brace_count
        prev_parent = skeleton(prev_parent).parent;
      end
      skeleton(nn).parent = prev_parent;
    end
    
    ii = ii+1;
            
  elseif strcmp( [C{ii},' ',C{ii+1}] , 'End Site' )
    % End effector; unnamed terminating joint
    %
    % N.B. The "two word" token here is why we don't use a switch statement
    % for this code.
    
    nn = nn+1;
    
    skeleton(nn).name = ' ';
    skeleton(nn).offset = [str2double(C(ii+4)) ; str2double(C(ii+5)) ; str2double(C(ii+6))];
    skeleton(nn).parent = nn-1; % always the direct child
    skeleton(nn).nestdepth = brace_count;
    skeleton(nn).Nchannels = 0;
        
  end
  
end

%% Initial processing and error checking

Nnodes = numel(skeleton);                       %�ܽڵ���
Nchannels = sum([skeleton.Nchannels]);          %��ͨ����
Nchainends = sum([skeleton.Nchannels]==0);      %��ȥEndsite����Ŀ

% Calculate number of header lines:
% ����ͷ�ļ�����Ŀ��һ���ڵ�Ϊ5�У��ڵ������������ţ�ƫ��ֵ��ͨ��ֵ���Ҳ�����ţ�
%һ��EndSite��4������һ��ͨ��ֵ
%5�ж����ֵ��һ�кͺ�����
%  - 5 lines per joint
%  - 4 lines per chain end
%  - 5 additional lines (first one and last four)
Nheaderlines = (Nnodes-Nchainends)*5 + Nchainends*4 + 5;

rawdata = importdata(fname,' ',Nheaderlines);

index = strncmp(rawdata.textdata,'Frames:',7);
Nframes = sscanf(rawdata.textdata{index},'Frames: %f');

index = strncmp(rawdata.textdata,'Frame Time:',10);
frame_time = sscanf(rawdata.textdata{index},'Frame Time: %f');

time = frame_time*(0:Nframes-1);

if size(rawdata.data,2) ~= Nchannels
  error('Error reading BVH file: channels count does not match.')
end

if size(rawdata.data,1) ~= Nframes
  warning('LOADBVH:frames_wrong','Error reading BVH file: frames count does not match; continuing anyway.')
end

%% Load motion data into skeleton structure
%
% We have three possibilities for each node we come across:
% (a) a root node that has displacements already defined,
%     for which the transformation matrix can be directly calculated;
% (b) a joint node, for which the transformation matrix must be calculated
%     from the previous points in the chain; and
% (c) an end effector, which only has displacement to calculate from the
%     previous node's transformation matrix and the offset of the end
%     joint.
%
% These are indicated in the skeleton structure, respectively, by having
% six, three, and zero "channels" of data.
% In this section of the code, the channels are read in where appropriate
% and the relevant arrays are pre-initialised for the subsequent calcs.

channel_count = 0;

for nn = 1:Nnodes
    
  if skeleton(nn).Nchannels == 6 % root node
    
    % assume translational data is always ordered XYZ
    skeleton(nn).Dxyz = repmat(skeleton(nn).offset,[1 Nframes]) + rawdata.data(:,channel_count+[1 2 3])'; %3*�ܶ�
    skeleton(nn).rxyz(skeleton(nn).order,:) = rawdata.data(:,channel_count+[4 5 6])';%3*�ܶ�
        
    % Kinematics of the root element:
    skeleton(nn).trans = nan(4,4,Nframes);
    for ff = 1:Nframes
      skeleton(nn).trans(:,:,ff) = transformation_matrix(skeleton(nn).Dxyz(:,ff) , skeleton(nn).rxyz(:,ff) , skeleton(nn).order);%����������ƫ�Ƶ���Ϣ����ת����Ϣ����ת�Ĵ���
    end
    
  elseif skeleton(nn).Nchannels == 3 % joint node
        
    skeleton(nn).rxyz(skeleton(nn).order,:) = rawdata.data(:,channel_count+[1 2 3])';
    skeleton(nn).Dxyz  = nan(3,Nframes);
    skeleton(nn).trans = nan(4,4,Nframes);
    
  elseif skeleton(nn).Nchannels == 0 % end node
    skeleton(nn).Dxyz  = nan(3,Nframes);
  end
  
  channel_count = channel_count + skeleton(nn).Nchannels;
  
end


%% Calculate kinematics
%
% No calculations are required for the root nodes.

% For each joint, calculate the transformation matrix and for convenience
% extract each position in a separate vector.
for nn = find([skeleton.parent] ~= 0 & [skeleton.Nchannels] ~= 0)
  
  parent = skeleton(nn).parent;
  
  for ff = 1:Nframes
    transM = transformation_matrix( skeleton(nn).offset , skeleton(nn).rxyz(:,ff) , skeleton(nn).order );%����ƫ�ƣ���ת��Ϣ����ת����
    skeleton(nn).trans(:,:,ff) = skeleton(parent).trans(:,:,ff) * transM;
    skeleton(nn).Dxyz(:,ff) = skeleton(nn).trans([1 2 3],4,ff);
  end

end

% For an end effector we don't have rotation data;
% just need to calculate the final position.
for nn = find([skeleton.Nchannels] == 0)
  
  parent = skeleton(nn).parent;
  
  for ff = 1:Nframes
    transM = skeleton(parent).trans(:,:,ff) * [eye(3), skeleton(nn).offset; 0 0 0 1];
    skeleton(nn).Dxyz(:,ff) = transM([1 2 3],4);
  end

end

end



function transM = transformation_matrix(displ,rxyz,order)
% Constructs the transformation for given displacement, DISPL, and
% rotations RXYZ. The vector RYXZ is of length three corresponding to
% rotations around the X, Y, Z axes.
%
% The third input, ORDER, is a vector indicating which order to apply
% the planar rotations. E.g., [3 1 2] refers applying rotations RXYZ
% around Z first, then X, then Y.
%
% Years ago we benchmarked that multiplying the separate rotation matrices
% was more efficient than pre-calculating the final rotation matrix
% symbolically, so we don't "optimise" by having a hard-coded rotation
% matrix for, say, 'ZXY' which seems more common in BVH files.
% Should revisit this assumption one day.
%
% Precalculating the cosines and sines saves around 38% in execution time.

c = cosd(rxyz);
s = sind(rxyz);

RxRyRz(:,:,1) = [1 0 0; 0 c(1) -s(1); 0 s(1) c(1)];
RxRyRz(:,:,2) = [c(2) 0 s(2); 0 1 0; -s(2) 0 c(2)];
RxRyRz(:,:,3) = [c(3) -s(3) 0; s(3) c(3) 0; 0 0 1];

rotM = RxRyRz(:,:,order(1))*RxRyRz(:,:,order(2))*RxRyRz(:,:,order(3));

transM = [rotM, displ; 0 0 0 1];

end
